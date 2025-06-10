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
from tqdm import tqdm

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
        """Train one epoch - FIXED VERSION without hanging."""
        print(f"[DEBUG] Process {self.accelerator.process_index}: Starting _train_epoch({epoch})")
        
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Create progress bar only for main process to avoid sync issues
        if self.accelerator.is_local_main_process:
            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}",
                leave=False
            )
            dataloader_iter = progress_bar
        else:
            dataloader_iter = self.dataloader
        
        print(f"[DEBUG] Process {self.accelerator.process_index}: Starting batch iteration for epoch {epoch}")
        
        for batch_idx, batch_data in enumerate(dataloader_iter):
            # Debug print for first few batches
            if batch_idx < 3 or batch_idx % 50 == 0:
                print(f"[DEBUG] Process {self.accelerator.process_index}: Epoch {epoch}, Batch {batch_idx}")
            
            # Forward pass
            try:
                outputs = self.model(**batch_data)
                loss = outputs.get('loss')
                
                if loss is None:
                    print(f"[DEBUG] Process {self.accelerator.process_index}: Epoch {epoch}, Batch {batch_idx}: Loss is None, skipping")
                    continue
                    
                if torch.isnan(loss):
                    print(f"[ERROR] Process {self.accelerator.process_index}: Epoch {epoch}, Batch {batch_idx}: NaN loss detected")
                    return float('nan')
                
            except Exception as e:
                print(f"[ERROR] Process {self.accelerator.process_index}: Forward pass error in epoch {epoch}, batch {batch_idx}: {e}")
                continue
            
            # Backward pass - CRITICAL: No sync operations here
            try:
                self.accelerator.backward(loss)
                
                # Gradient clipping (no sync needed)
                if self.clip_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Optimizer step (no sync needed)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            except Exception as e:
                print(f"[ERROR] Process {self.accelerator.process_index}: Backward pass error in epoch {epoch}, batch {batch_idx}: {e}")
                continue
            
            # Track metrics
            batch_loss_item = loss.item()
            epoch_loss += batch_loss_item
            num_batches += 1
            
            # Update progress bar only on main process
            if self.accelerator.is_local_main_process and hasattr(dataloader_iter, 'set_postfix'):
                dataloader_iter.set_postfix({"loss": f"{batch_loss_item:.4f}"})
            
            # Log at specified intervals (no sync operations)
            if (batch_idx + 1) % self.log_interval == 0 and self.accelerator.is_local_main_process:
                batch_size = batch_data.get('input_ids', next(iter(batch_data.values()))).shape[0]
                samples_processed = (batch_idx + 1) * batch_size
                print(f"[DEBUG] Process {self.accelerator.process_index}: Logging batch {batch_idx + 1}, loss: {batch_loss_item:.6f}")
            
            # CRITICAL: Remove any callback triggers that might contain sync operations
            # Only trigger callbacks on main process to avoid sync issues
            if self.accelerator.is_main_process:
                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': batch_loss_item})
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
        
        print(f"[DEBUG] Process {self.accelerator.process_index}: Completed _train_epoch({epoch}), avg_loss: {avg_loss:.6f}, num_batches: {num_batches}")
        
        return avg_loss
    
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