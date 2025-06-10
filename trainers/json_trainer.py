# trainers/json_trainer.py
"""
Simplified integration helpers for JSON logging - FIXED VERSION.
Removes complex wrapper conflicts and ensures logging works for all epochs.
"""

import logging
from typing import Dict, Any, Optional
from utils.json_logger import JSONLogger

logger = logging.getLogger(__name__)


class JSONLoggingAccelerateTrainer:
    """Simplified wrapper around AccelerateTrainer for JSON logging - FIXED."""
    
    def __init__(self, accelerate_trainer, json_logger: Optional[JSONLogger] = None):
        self.trainer = accelerate_trainer
        self.json_logger = json_logger
        self.accelerator = accelerate_trainer.accelerator
    
    def train(self):
        """Training loop with JSON logging - FIXED to work for all epochs."""
        
        # Store original methods once
        original_log_batch = self.trainer.log_batch
        original_log_epoch = self.trainer.log_epoch
        
        def enhanced_log_batch(batch_idx: int, loss: float, epoch: Optional[int] = None, metrics: Optional[Dict[str, Any]] = None):
            """Enhanced batch logging with JSON support."""
            # Call original logging first
            original_log_batch(batch_idx, loss, epoch, metrics)
            
            # Add JSON logging (only on main process and at specified intervals)
            if (self.json_logger and 
                self.accelerator.is_main_process and 
                batch_idx % self.json_logger.log_every_n_steps == 0):
                
                batch_metrics = {'loss': loss}
                if metrics:
                    batch_metrics.update(metrics)
                
                # Add process info
                batch_metrics.update({
                    'num_processes': self.accelerator.num_processes,
                    'is_main_process': True
                })
                
                self.json_logger.log_batch(
                    epoch=epoch or 0,
                    batch=batch_idx,
                    step=metrics.get('global_batch', batch_idx) if metrics else batch_idx,
                    metrics=batch_metrics
                )
        
        def enhanced_log_epoch(epoch: int, avg_loss: float, metrics: Optional[Dict[str, Any]] = None):
            """Enhanced epoch logging with JSON support."""
            # Call original logging first
            original_log_epoch(epoch, avg_loss, metrics)
            
            # Add JSON logging (only on main process)
            if self.json_logger and self.accelerator.is_main_process:
                epoch_metrics = {'loss': avg_loss}
                if metrics:
                    epoch_metrics.update(metrics)
                
                # Add process info
                epoch_metrics.update({
                    'num_processes': self.accelerator.num_processes,
                })
                                
                self.json_logger.log_epoch_end(epoch, epoch_metrics)
        
        # Replace methods
        self.trainer.log_batch = enhanced_log_batch
        self.trainer.log_epoch = enhanced_log_epoch
        
        # Log initial config
        if self.json_logger and self.accelerator.is_main_process:
            config_data = {
                'num_epochs': self.trainer.num_epochs,
                'batch_size': getattr(self.trainer.dataloader, 'batch_size', None),
                'num_processes': self.accelerator.num_processes,
                'mixed_precision': str(self.accelerator.mixed_precision),
                'device': str(self.accelerator.device),
                'log_interval': getattr(self.trainer, 'log_interval', 10)
            }
            self.json_logger.log_config(config_data)
        
        # Run training
        return self.trainer.train()
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped trainer."""
        return getattr(self.trainer, name)


def create_accelerate_trainer_with_json_logging(accelerate_trainer, json_logger: Optional[JSONLogger] = None):
    """Factory function to create JSON-enabled accelerate trainer."""
    return JSONLoggingAccelerateTrainer(accelerate_trainer, json_logger)