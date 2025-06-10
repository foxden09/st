"""
Clean wrapper for AccelerateTrainer with JSON logging.
FIXED: Now properly inherits from AccelerateTrainer with correct signature.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from .accelerate_trainer import AccelerateTrainer
from utils.json_logger import JSONLogger, create_json_logger_for_training

logger = logging.getLogger(__name__)


class AccelerateTrainerWithJSON(AccelerateTrainer):
    """
    Enhanced AccelerateTrainer with built-in JSON logging and validation.
    FIXED: Now properly matches parent class signature.
    """
    
    def __init__(self, 
                 model, 
                 dataloader, 
                 optimizer, 
                 device,  # ✅ FIXED: Use device, not accelerator
                 num_epochs: int = 10,
                 log_interval: int = 10,
                 output_dir: Optional[str] = None,
                 callbacks=None,
                 # JSON logging parameters
                 json_logger: Optional[JSONLogger] = None,
                 # Validation parameters
                 val_dataloader: Optional[DataLoader] = None,
                 validate_every: int = 1,
                 validate_every_n_batches: int = None):
        
        # ✅ FIXED: Pass device to parent, which creates accelerator internally
        super().__init__(
            model, dataloader, optimizer, device, 
            num_epochs, output_dir, callbacks
        )
        
        # Override num_epochs and log_interval from parent
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        
        # JSON logging and validation parameters
        self.json_logger = json_logger
        self.val_dataloader = val_dataloader
        self.validate_every = validate_every
        self.validate_every_n_batches = validate_every_n_batches
        
        # Global step counter
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # ✅ self.accelerator is now available from parent class
    
    def _run_validation(self) -> Dict[str, Any]:
        """
        Internal validation function to avoid import conflicts.
        """
        if not self.val_dataloader:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in self.val_dataloader:
                # Move to device
                batch = {k: v.to(self.accelerator.device) for k, v in batch_data.items() 
                        if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
                
                if loss is not None and not torch.isnan(loss):
                    batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
        
        self.model.train()  # Return to training mode
        
        if total_samples == 0:
            logger.warning("No validation samples processed")
            return {'loss': float('inf'), 'perplexity': float('inf'), 'samples': 0}
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'samples': total_samples
        }
    
    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch with JSON logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.dataloader)
        is_main_process = self.accelerator.is_main_process
        
        for batch_idx, batch in enumerate(self.dataloader):
            self._trigger_callbacks('on_batch_begin', batch_idx, logs={'epoch': epoch})
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.get('loss', outputs) if isinstance(outputs, dict) else outputs
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.clip_grad_norm:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Accumulate loss
            loss_value = loss.item()
            total_loss += loss_value
            self.global_step += 1
            
            # Logging
            if batch_idx % self.log_interval == 0 and is_main_process:
                self.log_batch(batch_idx, loss_value, epoch)
                
                # JSON logging
                if self.json_logger:
                    try:
                        perplexity = torch.exp(torch.tensor(loss_value)).item() if loss_value < 10 else float('inf')
                    except (RuntimeError, ValueError):
                        perplexity = float('inf')
                    
                    self.json_logger.log_batch(
                        epoch=epoch,
                        batch=batch_idx,
                        step=self.global_step,
                        metrics={  # Put loss and perplexity in metrics dict
                            'loss': loss_value,
                            'perplexity': perplexity
                        }
                    )
            
            # Mid-epoch validation
            if (self.val_dataloader and self.validate_every_n_batches and 
                batch_idx % self.validate_every_n_batches == 0 and 
                batch_idx > 0 and is_main_process):
                
                val_metrics = self._run_validation()
                if self.json_logger and val_metrics:
                    self.json_logger.log_validation(
                        epoch=epoch,
                        loss=val_metrics.get('loss'),
                        perplexity=val_metrics.get('perplexity'),
                        metrics={'batch_idx': batch_idx}
                    )
            
            self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': loss_value, 'epoch': epoch})
        
        return total_loss / num_batches

    def train(self) -> Dict[str, Any]:
        """Training loop with JSON logging and validation."""
        # Only log on main process
        is_main_process = self.accelerator.is_main_process
        
        if is_main_process:
            logger.info("Starting training...")
            
            # Log initial config
            if self.json_logger:
                config = {
                    'num_epochs': self.num_epochs,
                    'batch_size': self.dataloader.batch_size,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'log_interval': self.log_interval,
                    'num_processes': self.accelerator.num_processes,
                    'mixed_precision': str(self.accelerator.mixed_precision),
                    'has_validation': self.val_dataloader is not None,
                    'validate_every': self.validate_every
                }
                self.json_logger.log_config(config)
        
        self.model.train()
        total_start_time = time.time()
        
        # Training metrics
        training_metrics = {
            'epoch_losses': [],
            'val_losses': [],
            'val_perplexities': [],
            'final_loss': float('nan'),
            'final_val_loss': float('nan'),
            'training_time': 0.0,
            'total_batches': 0
        }
        
        self._trigger_callbacks('on_train_begin', logs={'num_epochs': self.num_epochs})
        
        for epoch in range(1, self.num_epochs + 1):
            if is_main_process:
                logger.info(f"Starting epoch {epoch}/{self.num_epochs}")
                
                # Log epoch start
                if self.json_logger:
                    self.json_logger.log_epoch_start(epoch)
            
            self._trigger_callbacks('on_epoch_begin', epoch)
            
            # Train one epoch
            epoch_start_time = time.time()
            epoch_loss = self._train_epoch(epoch)
            epoch_duration = time.time() - epoch_start_time
            
            training_metrics['epoch_losses'].append(epoch_loss)
            
            # Run validation if needed (only on main process)
            val_loss = None
            val_perplexity = None
            if self.val_dataloader and (epoch % self.validate_every == 0) and is_main_process:
                logger.info(f"Running validation for epoch {epoch}...")
                val_metrics = self._run_validation()
                val_loss = val_metrics.get('loss')
                val_perplexity = val_metrics.get('perplexity')
                
                if val_loss is not None:
                    training_metrics['val_losses'].append(val_loss)
                    training_metrics['val_perplexities'].append(val_perplexity)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.output_dir:
                            os.makedirs(self.output_dir, exist_ok=True)
                            best_path = os.path.join(self.output_dir, "best_model.pt")
                            self.save_checkpoint_fixed(best_path, epoch=epoch, val_loss=val_loss, is_best=True)
                            logger.info(f"New best validation loss: {val_loss:.4f}")
                else:
                    logger.warning("Validation returned None loss value")

            # Epoch logging
            epoch_metrics = {
                'train_loss': epoch_loss,
                'epoch_duration': epoch_duration
            }
            if val_loss is not None:
                epoch_metrics['val_loss'] = val_loss
                epoch_metrics['val_perplexity'] = val_perplexity
            
            if is_main_process:
                self.log_epoch(epoch, epoch_loss, epoch_metrics)
                
                # JSON epoch logging
                if self.json_logger:
                    self.json_logger.log_epoch_end(epoch, epoch_metrics)
                    
                    # Also log validation if available
                    if val_loss is not None:
                        self.json_logger.log_validation(epoch, {
                            'loss': val_loss,
                            'perplexity': val_perplexity
                        })
            
            self._trigger_callbacks('on_epoch_end', epoch, logs=epoch_metrics)
            
            # Save checkpoint
            if self.output_dir and is_main_process:
                os.makedirs(self.output_dir, exist_ok=True)
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint_fixed(checkpoint_path, epoch=epoch)
        
        # Training complete
        total_time = time.time() - total_start_time
        training_metrics['training_time'] = total_time
        training_metrics['total_batches'] = self.global_step
        
        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        if training_metrics['val_losses']:
            training_metrics['final_val_loss'] = training_metrics['val_losses'][-1]
            training_metrics['final_val_perplexity'] = training_metrics['val_perplexities'][-1]
        
        if is_main_process:
            logger.info(f"Training completed in {total_time:.2f} seconds")
            logger.info(f"Final loss: {training_metrics['final_loss']:.4f}")
            if training_metrics['val_losses']:
                logger.info(f"Final validation loss: {training_metrics['final_val_loss']:.4f}")
        
        self._trigger_callbacks('on_train_end', logs=training_metrics)
        
        return training_metrics

    def save_checkpoint_fixed(self, path: str, epoch: int, val_loss: float = None, is_best: bool = False):
        """Save checkpoint with proper unwrapping."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Get the unwrapped model
        model_to_save = self.accelerator.unwrap_model(self.model)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }
        
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
            
        if is_best:
            checkpoint['is_best'] = True
            
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")


def create_accelerate_trainer_with_json_logging(
    model, dataloader, optimizer, device,  # ✅ FIXED: Use device parameter
    output_dir: str,
    experiment_name: str = "training",
    num_epochs: int = 10,
    log_interval: int = 10,
    json_log_every_n_steps: int = 50,
    val_dataloader=None,
    validate_every: int = 1,
    validate_every_n_batches: int = None,
    **kwargs  # Handle any extra parameters
) -> AccelerateTrainerWithJSON:
    """Create AccelerateTrainer with JSON logging setup."""
    
    # Create JSON logger
    json_logger = create_json_logger_for_training(
        output_dir=output_dir,
        experiment_name=experiment_name,
        log_every_n_steps=json_log_every_n_steps
    )
    
    return AccelerateTrainerWithJSON(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,  # ✅ FIXED: Pass device
        num_epochs=num_epochs,
        log_interval=log_interval,
        output_dir=output_dir,
        json_logger=json_logger,
        val_dataloader=val_dataloader,
        validate_every=validate_every,
        validate_every_n_batches=validate_every_n_batches
    )