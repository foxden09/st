"""
Clean wrapper for AccelerateTrainer with JSON logging.
Fixes the logging issues by properly integrating with the training loop.
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
from utils.validation_utils import run_validation

logger = logging.getLogger(__name__)


class AccelerateTrainerWithJSON(AccelerateTrainer):
    """
    Enhanced AccelerateTrainer with built-in JSON logging and validation.
    """
    
    def __init__(self, 
                 model, 
                 dataloader, 
                 optimizer, 
                 accelerator,
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
        
        super().__init__(model, dataloader, optimizer, accelerator, output_dir, callbacks)
        
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.json_logger = json_logger
        self.val_dataloader = val_dataloader
        self.validate_every = validate_every
        self.validate_every_n_batches = validate_every_n_batches
        
        # Global step counter
        self.global_step = 0

        self.best_val_loss = float('inf')
    
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
                val_metrics = run_validation(self.model, self.val_dataloader, self.accelerator.device)
                val_loss = val_metrics['loss']
                val_perplexity = val_metrics['perplexity']
                
                training_metrics['val_losses'].append(val_loss)
                training_metrics['val_perplexities'].append(val_perplexity)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(self.output_dir, "best_model.pt")
                    self.save_checkpoint_fixed(best_path, epoch=epoch, val_loss=val_loss, is_best=True)
                    logger.info(f"New best validation loss: {val_loss:.4f}")

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
                
                # Log epoch end to JSON
                if self.json_logger:
                    self.json_logger.log_epoch_end(epoch, epoch_metrics)
            
            self._trigger_callbacks('on_epoch_end', epoch, logs=epoch_metrics)
            
            # Save checkpoint (only on main process)
            if self.output_dir and is_main_process:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path, epoch=epoch, loss=epoch_loss)
        
        # Final metrics
        training_metrics['training_time'] = time.time() - total_start_time
        training_metrics['total_batches'] = self.global_step
        
        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        if training_metrics['val_losses']:
            training_metrics['final_val_loss'] = training_metrics['val_losses'][-1]
        
        if is_main_process:
            logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
            logger.info(f"Final training loss: {training_metrics['final_loss']:.6f}")
            if training_metrics['final_val_loss'] != float('nan'):
                logger.info(f"Final validation loss: {training_metrics['final_val_loss']:.6f}")
        
        self._trigger_callbacks('on_train_end', logs=training_metrics)
        
        training_metrics['best_val_loss'] = getattr(self, 'best_val_loss', float('inf'))
        
        return training_metrics
    
    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch with proper JSON logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        is_main_process = self.accelerator.is_main_process
        
        for batch_idx, batch in enumerate(self.dataloader):
            self.global_step += 1
            
            # Move batch to device (accelerator handles this)
            input_ids = batch['input_ids']
            labels = batch.get('labels', input_ids)
            
            # Forward pass
            with self.accelerator.accumulate(self.model):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Backward pass
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Calculate perplexity for this batch
            batch_perplexity = torch.exp(loss).item()
            
            # Batch logging (only on main process, every log_interval batches)
            if is_main_process and batch_idx % self.log_interval == 0:
                self.log_batch(batch_idx, batch_loss, epoch, metrics={'perplexity': batch_perplexity})
            
            # JSON batch logging (only on main process, every 50 batches)
            if self.json_logger and is_main_process and batch_idx % 50 == 0:
                self.json_logger.log_batch(epoch, batch_idx, self.global_step, batch_loss, batch_perplexity)
            
            # Run mini-validation every N batches if configured
            if (self.validate_every_n_batches and 
                self.val_dataloader and 
                batch_idx > 0 and 
                batch_idx % self.validate_every_n_batches == 0 and
                is_main_process):
                
                logger.info(f"Running mini-validation at batch {batch_idx}...")
                val_metrics = run_validation(self.model, self.val_dataloader, self.accelerator.device, max_batches=10)
                val_loss = val_metrics['loss']
                val_perplexity = val_metrics['perplexity']
                
                logger.info(f"Mini-validation - Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
                
                # Log mini-validation to JSON
                if self.json_logger:
                    self.json_logger.log_validation(epoch, val_loss, val_perplexity, 
                                                   metrics={'type': 'mini_validation', 'batch': batch_idx})
                
                # Reset model to training mode
                self.model.train()
            
            self._trigger_callbacks('on_batch_end', batch_idx, logs={
                'loss': batch_loss,
                'perplexity': batch_perplexity,
                'epoch': epoch,
                'global_step': self.global_step
            })
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def log_batch(self,
                  batch_idx: int,
                  loss: float,
                  epoch: Optional[int] = None,
                  metrics: Optional[Dict[str, Any]] = None):
        """Log information about a training batch with perplexity."""
        # Build metrics string
        metrics_parts = []
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, float):
                    metrics_parts.append(f"{k}: {v:.4f}")
                else:
                    metrics_parts.append(f"{k}: {v}")
        
        metrics_str = ", ".join(metrics_parts)
        epoch_str = f"Epoch {epoch}, " if epoch is not None else ""
        
        logger.info(f"{epoch_str}Batch {batch_idx}, Loss: {loss:.4f}" +
                   (f", {metrics_str}" if metrics_str else ""))


def create_accelerate_trainer_with_json_logging(
    model, dataloader, optimizer, accelerator,
    output_dir: str,
    experiment_name: str = "training",
    num_epochs: int = 10,
    log_interval: int = 10,
    json_log_every_n_steps: int = 50,
    val_dataloader=None,
    validate_every: int = 1,
    validate_every_n_batches: int = None
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
        accelerator=accelerator,
        num_epochs=num_epochs,
        log_interval=log_interval,
        output_dir=output_dir,
        json_logger=json_logger,
        val_dataloader=val_dataloader,
        validate_every=validate_every,
        validate_every_n_batches=validate_every_n_batches
    )
