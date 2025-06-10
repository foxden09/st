#!/usr/bin/env python
"""
Plot training loss over batches from JSON log files.
Usage: python plot_json_loss.py path/to/logfile.jsonl
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Tuple, Optional
import numpy as np


def extract_batch_losses(json_file_path: str) -> Tuple[List[int], List[float], List[int]]:
    """
    Extract batch numbers, losses, and epoch boundaries from JSON log file.
    
    Args:
        json_file_path: Path to the .jsonl file
        
    Returns:
        (batch_numbers, losses, epoch_boundaries) where epoch_boundaries are batch numbers where epochs end
    """
    batch_numbers = []
    losses = []
    epoch_boundaries = []
    
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return [], [], []
    
    print(f"Reading JSON log file: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                event = json.loads(line)
                event_type = event.get('event_type', '')
                
                # Extract batch loss data
                if event_type == 'batch':
                    step = event.get('step', 0)
                    metrics = event.get('metrics', {})
                    loss = metrics.get('loss')
                    
                    if loss is not None and isinstance(loss, (int, float)):
                        batch_numbers.append(step)
                        losses.append(float(loss))
                
                # Extract epoch boundaries
                elif event_type == 'epoch_end':
                    epoch = event.get('epoch', 0)
                    metrics = event.get('metrics', {})
                    global_batch = metrics.get('global_batch', 0)
                    
                    if global_batch > 0:
                        epoch_boundaries.append(global_batch)
                        print(f"Found epoch {epoch} ending at batch {global_batch}")
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"Extracted {len(batch_numbers)} batch loss points")
    print(f"Found {len(epoch_boundaries)} epoch boundaries")
    
    return batch_numbers, losses, epoch_boundaries


def plot_training_loss(batch_numbers: List[int], 
                      losses: List[float], 
                      epoch_boundaries: List[int],
                      json_file_path: str,
                      save_path: Optional[str] = None,
                      show_plot: bool = True,
                      smooth_window: int = 50) -> None:
    """
    Create training loss plot with epoch boundaries and smoothing.
    
    Args:
        batch_numbers: List of batch numbers
        losses: List of corresponding losses
        epoch_boundaries: List of batch numbers where epochs end
        json_file_path: Original JSON file path (for title)
        save_path: Where to save the plot (optional)
        show_plot: Whether to display the plot
        smooth_window: Window size for smoothing (0 = no smoothing)
    """
    if not batch_numbers or not losses:
        print("No loss data to plot!")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Raw loss data
    ax1 = axes[0]
    ax1.plot(batch_numbers, losses, alpha=0.7, linewidth=1, color='blue', label='Training Loss')
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss Over Batches\nFile: {os.path.basename(json_file_path)}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add epoch boundaries
    for i, boundary in enumerate(epoch_boundaries):
        if boundary <= max(batch_numbers):
            ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.6, linewidth=1)
            # Add epoch labels
            y_pos = max(losses) * 0.9 if losses else 1.0
            ax1.text(boundary, y_pos, f'E{i+1}', rotation=90, 
                    verticalalignment='bottom', fontsize=9, color='red')
    
    # Plot 2: Smoothed loss (if enough data points)
    ax2 = axes[1]
    if len(losses) > smooth_window and smooth_window > 0:
        # Calculate moving average
        smoothed_losses = []
        smoothed_batches = []
        
        for i in range(len(losses)):
            start_idx = max(0, i - smooth_window // 2)
            end_idx = min(len(losses), i + smooth_window // 2 + 1)
            
            window_losses = losses[start_idx:end_idx]
            smoothed_loss = sum(window_losses) / len(window_losses)
            
            smoothed_losses.append(smoothed_loss)
            smoothed_batches.append(batch_numbers[i])
        
        ax2.plot(smoothed_batches, smoothed_losses, linewidth=2, color='orange', 
                label=f'Smoothed Loss (window={smooth_window})')
        ax2.set_title(f'Smoothed Training Loss (Moving Average)')
    else:
        # Not enough data for smoothing, just plot raw data
        ax2.plot(batch_numbers, losses, linewidth=2, color='orange', label='Training Loss')
        ax2.set_title('Training Loss (No Smoothing - Insufficient Data)')
    
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add epoch boundaries to smoothed plot too
    for boundary in epoch_boundaries:
        if boundary <= max(batch_numbers):
            ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nTraining Summary:")
    print(f"Total batches: {max(batch_numbers) if batch_numbers else 0}")
    print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {min(losses):.4f}")
    print(f"Epochs completed: {len(epoch_boundaries)}")
    
    if epoch_boundaries and batch_numbers:
        avg_batches_per_epoch = max(batch_numbers) / len(epoch_boundaries)
        print(f"Average batches per epoch: {avg_batches_per_epoch:.1f}")
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Plot training loss from JSON log files')
    parser.add_argument('json_file', help='Path to the JSON log file (.jsonl)')
    parser.add_argument('--save', type=str, help='Path to save the plot (e.g., plot.png)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display the plot')
    parser.add_argument('--smooth', type=int, default=50, 
                       help='Smoothing window size (default: 50, 0 = no smoothing)')
    
    args = parser.parse_args()
    
    # Extract data from JSON file
    batch_numbers, losses, epoch_boundaries = extract_batch_losses(args.json_file)
    
    if not batch_numbers:
        print("No training data found in the JSON file!")
        return
    
    # Generate save path if not provided
    save_path = args.save
    if save_path is None:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        save_path = f"{base_name}_loss_plot.png"
    
    # Create the plot
    plot_training_loss(
        batch_numbers=batch_numbers,
        losses=losses,
        epoch_boundaries=epoch_boundaries,
        json_file_path=args.json_file,
        save_path=save_path,
        show_plot=not args.no_show,
        smooth_window=args.smooth
    )


if __name__ == "__main__":
    main()