#!/bin/bash

# 1. Original training (1 epoch)
python examples/train_symbolic_example.py \
  --use_proj --use_v \
  --batch_size 8 --effective_batch_size 8 \
  --num_epochs 1 --max_samples 100000 \
  --output_dir "./outputs/symbolic_test"

# 2. Resume with 2x effective batch (epoch 2)
python examples/train_symbolic_example.py \
  --use_proj --use_v \
  --resume_from_checkpoint "./outputs/symbolic_test/checkpoint_epoch_1.pt" \
  --batch_size 8 --effective_batch_size 32 \
  --num_epochs 3 --max_samples 100000 \
  --output_dir "./outputs/symbolic_test"

# 3. Resume with same effective batch (epoch 3)
python examples/train_symbolic_example.py \
  --use_proj --use_v \
  --resume_from_checkpoint "./outputs/symbolic_test/checkpoint_epoch_3.pt" \
  --batch_size 8 --effective_batch_size 64 \
  --num_epochs 8 --max_samples 100000 \
  --output_dir "./outputs/symbolic_test"

