"""
Enhanced validation utilities with support for max_batches parameter.
"""

import torch
import logging
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_validation(model: torch.nn.Module, 
                  val_dataloader: DataLoader, 
                  device: torch.device,
                  max_batches: Optional[int] = None) -> Dict[str, Any]:
    """
    Run validation on the model with optional batch limiting for mini-validation.
    
    Args:
        model: The model to validate
        val_dataloader: Validation data loader
        device: Device to run validation on
        max_batches: Optional limit on number of batches for quick validation
        
    Returns:
        Dictionary with validation metrics (loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Stop early if max_batches is specified (for mini-validation)
            if max_batches and batch_idx >= max_batches:
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids).to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            total_loss += loss.item()
            total_batches += 1
    
    if total_batches == 0:
        logger.warning("No validation batches processed")
        return {'loss': float('inf'), 'perplexity': float('inf')}
    
    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': total_batches
    }
