"""
Simplified JSON logging utility for training metrics.
Clean, straightforward implementation without overcomplication.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging


class JSONLogger:
    """Simple JSON logger for training metrics."""
    
    def __init__(self, log_file: str, experiment_name: str = None, log_every_n_steps: int = 100):
        self.log_file = log_file
        self.experiment_name = experiment_name or "experiment"
        self.log_every_n_steps = log_every_n_steps
        self.start_time = time.time()
        self.step_count = 0
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize log file
        self._log_event("experiment_start", {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "log_every_n_steps": self.log_every_n_steps
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Write a JSON event to the log file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "event_type": event_type,
            **data
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            self.logger.warning(f"Failed to write JSON log: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        self._log_event("config", {"config": config})
    
    def log_epoch_start(self, epoch: int):
        """Log epoch start."""
        self._log_event("epoch_start", {"epoch": epoch})
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch completion."""
        self._log_event("epoch_end", {
            "epoch": epoch,
            "metrics": metrics
        })
    
    def log_batch(self, epoch: int, batch: int, step: int, loss: float, perplexity: float = None, metrics: Dict[str, Any] = None):
        """Log batch metrics at specified intervals."""
        self.step_count = step
        if batch % self.log_every_n_steps == 0:  # Use batch number instead of step
            batch_data = {
                "epoch": epoch,
                "batch": batch,
                "step": step,
                "loss": loss
            }
            if perplexity is not None:
                batch_data["perplexity"] = perplexity
            if metrics:
                batch_data["metrics"] = metrics
            self._log_event("batch", batch_data)
    
    def log_validation(self, epoch: int, loss: float, perplexity: float, metrics: Dict[str, Any] = None):
        """Log validation metrics."""
        val_data = {
            "epoch": epoch,
            "loss": loss,
            "perplexity": perplexity
        }
        if metrics:
            val_data.update(metrics)
        self._log_event("validation", val_data)


def create_json_logger_for_training(output_dir: str, experiment_name: str, log_every_n_steps: int = 50) -> JSONLogger:
    """Create JSON logger for training runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, 'logs', f'{experiment_name}_{timestamp}.jsonl')
    return JSONLogger(log_file, experiment_name, log_every_n_steps)
