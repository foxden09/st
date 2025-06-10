#!/bin/bash
# =============================================================================
# FILE 1: scripts/vanilla_train_acc.sh (FIXED)
# =============================================================================
# Enhanced Vanilla Transformer Training with 4 GPUs using new logging system
# Baseline comparison to symbolic transformer with same parameters

set -e  # Exit on any error

# Configuration - matching symbolic script parameters
DIR="./outputs/vanilla_4gpu_enhanced"
N=110000
EXPERIMENT_NAME="vanilla_4gpu_enhanced"

# Model configuration - matching symbolic script
N_EMBD=384
PRESET="small"

# Multi-GPU configuration - matching symbolic script
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Enhanced batch configuration (using new logging system)
BATCH_SIZE=4  # Direct batch size per GPU

# Enhanced JSON logging configuration
JSON_LOG_STEPS=50  # Log every 50 batches (matching new system)
LOG_INTERVAL=10    # Console logs every 10 batches

# Validation configuration
VAL_RATIO=0.1                    # 10% for validation
VALIDATE_EVERY=1                 # Full validation every epoch
VALIDATE_EVERY_N_BATCHES=100     # Mini-validation every 100 batches

echo "========================================================"
echo "ENHANCED VANILLA TRANSFORMER 4-GPU TRAINING (BASELINE)"
echo "========================================================"
echo "Output directory: $DIR"
echo "Max samples: $N"
echo "Number of GPUs: $NUM_GPUS"
echo "Model size: $N_EMBD dimensions"
echo "Enhanced JSON logging: Every $JSON_LOG_STEPS batches"
echo "Console logging: Every $LOG_INTERVAL batches"
echo "Mini-validation: Every $VALIDATE_EVERY_N_BATCHES batches"
echo "Experiment name: $EXPERIMENT_NAME"
echo ""
echo "Batch size: $BATCH_SIZE per GPU (${BATCH_SIZE}×4 = $((BATCH_SIZE * 4)) total)"
echo "Validation split: ${VAL_RATIO} (${N%.*}*${VAL_RATIO} samples)"
echo ""
echo "Key features (Enhanced Baseline for comparison):"
echo "  ✓ Standard transformer architecture with positional embeddings"
echo "  ✓ Enhanced JSON logging with batch-level perplexity"
echo "  ✓ Mini-validation every N batches for real-time monitoring"
echo "  ✓ Training and validation perplexity tracking"
echo "  ✓ Clean JSON logs with structured batch tracking"
echo "  ✓ No gradient accumulation complexity"
echo "========================================================"

# Create output directory
mkdir -p $DIR

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available or Python/PyTorch not working"
    exit 1
fi

# Single training run - all 8 epochs using enhanced logging
echo ""
echo "========================================================"
echo "ENHANCED TRAINING: All 8 epochs (Vanilla Transformer)"
echo "Using new enhanced logging system with validation"
echo "Batch size: $BATCH_SIZE per GPU (${BATCH_SIZE}×4 = $((BATCH_SIZE * 4)) total)"
echo "========================================================"

accelerate launch \
    --config_file ./accelerate_config_4gpu.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    examples/train_vanilla_with_json_logging.py \
    --preset $PRESET \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --num_epochs 8 \
    --max_samples $N \
    --output_dir $DIR \
    --use_accelerate \
    --json_log_steps $JSON_LOG_STEPS \
    --log_interval $LOG_INTERVAL \
    --experiment_name $EXPERIMENT_NAME \
    --val_ratio $VAL_RATIO \
    --validate_every $VALIDATE_EVERY \
    --validate_every_n_batches $VALIDATE_EVERY_N_BATCHES \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --generation_prompts "Once upon a time" "The little girl" "A brave knight"

if [ $? -ne 0 ]; then
    echo "Enhanced training failed. Exiting."
    exit 1
fi

echo "Enhanced training completed successfully!"

# Generate enhanced training plots from JSON logs
echo ""
echo "========================================================"
echo "GENERATING ENHANCED TRAINING PLOTS FROM JSON LOGS"
echo "========================================================"

python -c "
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print('Starting enhanced plot generation...')

# Find all JSON log files
log_dir = '$DIR/logs'
json_files = []
if os.path.exists(log_dir):
    for f in os.listdir(log_dir):
        if f.endswith('.jsonl') and '$EXPERIMENT_NAME' in f:
            json_files.append(os.path.join(log_dir, f))

print(f'Found {len(json_files)} JSON log files')

# Extract enhanced training metrics
batch_data = []
epoch_data = []
validation_data = []
mini_validation_data = []

for json_file in sorted(json_files):
    print(f'Processing: {json_file}')
    with open(json_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                
                # Batch-level data with perplexity
                if event.get('event_type') == 'batch':
                    batch_data.append({
                        'step': event.get('step', 0),
                        'epoch': event.get('epoch', 0),
                        'batch': event.get('batch', 0),
                        'loss': event.get('loss'),
                        'perplexity': event.get('perplexity')
                    })
                
                # Epoch-level data
                elif event.get('event_type') == 'epoch_end':
                    metrics = event.get('metrics', {})
                    epoch_data.append({
                        'epoch': event.get('epoch', 0),
                        'train_loss': metrics.get('train_loss'),
                        'val_loss': metrics.get('val_loss'),
                        'val_perplexity': metrics.get('val_perplexity'),
                        'duration': metrics.get('epoch_duration')
                    })
                
                # Validation data
                elif event.get('event_type') == 'validation':
                    val_type = event.get('type', 'full_validation')
                    data_point = {
                        'epoch': event.get('epoch', 0),
                        'loss': event.get('loss'),
                        'perplexity': event.get('perplexity'),
                        'type': val_type
                    }
                    
                    if val_type == 'mini_validation':
                        data_point['batch'] = event.get('batch', 0)
                        mini_validation_data.append(data_point)
                    else:
                        validation_data.append(data_point)
                        
            except (json.JSONDecodeError, KeyError) as e:
                continue

print(f'Extracted {len(batch_data)} batch points, {len(epoch_data)} epochs, {len(validation_data)} validations, {len(mini_validation_data)} mini-validations')

# Create enhanced plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Enhanced Vanilla Transformer Training Progress (Baseline)', fontsize=16)

# Plot 1: Batch-level training loss and perplexity
if batch_data:
    steps = [d['step'] for d in batch_data if d['loss'] is not None]
    losses = [d['loss'] for d in batch_data if d['loss'] is not None]
    perplexities = [d['perplexity'] for d in batch_data if d['perplexity'] is not None]
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(steps, losses, 'b-', alpha=0.7, linewidth=1, label='Training Loss')
    line2 = ax1_twin.plot(steps, perplexities, 'r-', alpha=0.7, linewidth=1, label='Training Perplexity')
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss', color='b')
    ax1_twin.set_ylabel('Perplexity', color='r')
    ax1.set_title('Batch-level Training Progress')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

# Plot 2: Epoch-level comparison (train vs validation)
if epoch_data and validation_data:
    epochs = [d['epoch'] for d in epoch_data]
    train_losses = [d['train_loss'] for d in epoch_data]
    val_losses = [d['val_loss'] for d in validation_data if d['loss'] is not None]
    val_epochs = [d['epoch'] for d in validation_data if d['loss'] is not None]
    
    ax2.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax2.plot(val_epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training vs Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 3: Validation perplexity over time
if validation_data and mini_validation_data:
    # Full validation perplexity
    val_epochs = [d['epoch'] for d in validation_data if d['perplexity'] is not None]
    val_perplexities = [d['perplexity'] for d in validation_data if d['perplexity'] is not None]
    
    # Mini-validation perplexity (convert to approximate step)
    mini_epochs = [d['epoch'] + d.get('batch', 0)/1000.0 for d in mini_validation_data if d['perplexity'] is not None]
    mini_perplexities = [d['perplexity'] for d in mini_validation_data if d['perplexity'] is not None]
    
    ax3.plot(val_epochs, val_perplexities, 'g-o', label='Full Validation', linewidth=2, markersize=8)
    if mini_validation_data:
        ax3.plot(mini_epochs, mini_perplexities, 'orange', alpha=0.7, marker='.', linestyle='-', 
                label='Mini-validation', linewidth=1, markersize=4)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Perplexity')
    ax3.set_title('Validation Perplexity Progress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot 4: Training efficiency (loss reduction rate)
if batch_data:
    # Calculate loss reduction rate over windows
    window_size = max(1, len(batch_data) // 20)  # 20 data points
    steps = []
    loss_rates = []
    
    for i in range(window_size, len(batch_data), window_size):
        if batch_data[i]['loss'] and batch_data[i-window_size]['loss']:
            start_loss = batch_data[i-window_size]['loss']
            end_loss = batch_data[i]['loss']
            rate = (start_loss - end_loss) / window_size  # Loss reduction per step
            
            steps.append(batch_data[i]['step'])
            loss_rates.append(rate)
    
    if loss_rates:
        ax4.plot(steps, loss_rates, 'purple', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Loss Reduction Rate')
        ax4.set_title('Training Efficiency (Loss Reduction Rate)')
        ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save enhanced plot
plot_path = '$DIR/enhanced_training_progress_vanilla_baseline.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f'Enhanced training plots saved to: {plot_path}')

# Save training summary
summary_path = '$DIR/enhanced_training_summary.txt'
with open(summary_path, 'w') as f:
    f.write('ENHANCED VANILLA TRANSFORMER TRAINING SUMMARY\\n')
    f.write('=' * 50 + '\\n')
    f.write(f'Experiment: $EXPERIMENT_NAME\\n')
    f.write(f'Training completed: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\\n\\n')
    
    if epoch_data:
        final_epoch = epoch_data[-1]
        f.write(f'Final Training Loss: {final_epoch[\"train_loss\"]:.4f}\\n')
        if final_epoch.get('val_loss'):
            f.write(f'Final Validation Loss: {final_epoch[\"val_loss\"]:.4f}\\n')
    
    if validation_data:
        final_val = validation_data[-1]
        f.write(f'Final Validation Perplexity: {final_val[\"perplexity\"]:.2f}\\n')
    
    f.write(f'\\nTotal Batch Steps: {len(batch_data)}\\n')
    f.write(f'Total Epochs: {len(epoch_data)}\\n')
    f.write(f'Validation Points: {len(validation_data)}\\n')
    f.write(f'Mini-validation Points: {len(mini_validation_data)}\\n')
    
    f.write(f'\\nEnhanced Features Used:\\n')
    f.write(f'- Batch-level perplexity tracking\\n')
    f.write(f'- Mini-validation every $VALIDATE_EVERY_N_BATCHES batches\\n')
    f.write(f'- JSON logging every $JSON_LOG_STEPS batches\\n')
    f.write(f'- Structured validation monitoring\\n')

print(f'Enhanced training summary saved to: {summary_path}')
print('Enhanced plot generation completed!')
"

if [ $? -ne 0 ]; then
    echo "Enhanced plot generation failed, but training was successful."
fi

echo ""
echo "========================================================"
echo "ENHANCED VANILLA TRANSFORMER TRAINING COMPLETE!"
echo "========================================================"
echo "Output files:"
echo "  Model checkpoints: $DIR/checkpoint_epoch_*.pt"
echo "  Enhanced JSON logs: $DIR/logs/"
echo "  Enhanced plots: $DIR/enhanced_training_progress_vanilla_baseline.png"
echo "  Training summary: $DIR/enhanced_training_summary.txt"
echo ""
echo "Enhanced baseline ready for comparison with symbolic transformer!"
echo "Key improvements: batch-level perplexity, mini-validation, structured logging"
echo "========================================================"

