# =============================================================================
# FILE 1: examples/train_vanilla_with_json_logging.py (FIXED)
# =============================================================================
#!/usr/bin/env python
"""
Fixed training script for Vanilla Transformer with enhanced JSON logging and validation.
Baseline comparison to symbolic transformer - clean implementation.

Usage:
    python examples/train_vanilla_with_json_logging.py --preset small --json_log_steps 50 --val_ratio 0.1
    python examples/train_vanilla_with_json_logging.py --preset medium --disable_json_logging --validate_every 2
    python examples/train_vanilla_with_json_logging.py --batch_size 32 --no_validation
"""

import argparse
import os
import sys
import torch
import logging
from datetime import datetime
from torch.utils.data import random_split

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import SymbolicConfig, get_preset_config, print_config
from mytokenizers import create_tokenizer
from model import get_model
from utils.data_utils import load_and_prepare_data
from inference.generation import run_generation

# Enhanced JSON logging imports
from trainers.simple_trainer_with_json import SimpleTrainerWithJSON
from trainers.accelerate_trainer_with_json import create_accelerate_trainer_with_json_logging, create_json_logger_for_training

# Suppress output on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0': 
    sys.stdout = open(os.devnull, 'w')

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments with enhanced JSON logging and validation support."""
    parser = argparse.ArgumentParser(description='Train Vanilla Transformer with Enhanced JSON Logging and Validation')
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=10000)
    
    # Model configuration
    parser.add_argument('--preset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--block_size', type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bias", action='store_true')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--log_interval', type=int, default=10)
    
    # Enhanced JSON logging parameters
    parser.add_argument('--json_log_steps', type=int, default=50, 
                       help='Log to JSON every N batches (default: 50)')
    parser.add_argument('--disable_json_logging', action='store_true',
                       help='Disable JSON logging completely')
    parser.add_argument('--experiment_name', type=str, default='vanilla_transformer',
                       help='Name for this experiment in logs')
    
    # Enhanced validation parameters
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Fraction of data to use for validation (default: 0.1)')
    parser.add_argument('--validate_every', type=int, default=1,
                       help='Run full validation every N epochs (default: 1)')
    parser.add_argument('--validate_every_n_batches', type=int, default=None,
                       help='Run mini-validation every N batches (optional)')
    parser.add_argument('--no_validation', action='store_true',
                       help='Disable validation completely')
    
    # Output and generation
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--generation_prompts', nargs='+', 
                       default=["Once upon a time", "The little girl"],
                       help='Prompts for text generation examples')
    parser.add_argument('--generate_every', type=int, default=1,
                       help='Generate text every N epochs')
    
    # Distributed training
    parser.add_argument('--use_accelerate', action='store_true',
                       help='Use Accelerate for distributed training')
    
    return parser.parse_args()


def create_vanilla_config(args):
    """Create configuration for the vanilla transformer."""
    # Start with preset config
    config = get_preset_config(args.preset)
    
    # Override with command line arguments
    overrides = {}
    if args.block_size is not None:
        overrides['block_size'] = args.block_size
    if args.n_layer is not None:
        overrides['n_layer'] = args.n_layer
    if args.n_head is not None:
        overrides['n_head'] = args.n_head
    if args.n_embd is not None:
        overrides['n_embd'] = args.n_embd
    if args.dropout is not None:
        overrides['dropout'] = args.dropout
    if args.bias:
        overrides['bias'] = True
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        overrides['weight_decay'] = args.weight_decay
    
    # Apply overrides
    for key, value in overrides.items():
        setattr(config, key, value)
    
    # Ensure vanilla transformer settings (no symbolic features)
    config.use_symbolic_ffn = False
    config.use_vocab_refinement = False
    config.use_v = False
    config.use_proj = False
    
    return config


def load_and_split_data(args, tokenizer):
    """Load data and create train/validation split."""
    # Load full dataset
    full_dataloader, _ = load_and_prepare_data(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_seq_length=getattr(args, 'block_size', 128),
        batch_size=getattr(args, 'batch_size', 32)
    )
    
    if args.no_validation:
        return full_dataloader, None
    
    # Split dataset
    dataset = full_dataloader.dataset
    train_size = int((1.0 - args.val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create separate dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=full_dataloader.batch_size,
        shuffle=True,
        collate_fn=full_dataloader.collate_fn,
        drop_last=True,
        num_workers=0
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=full_dataloader.batch_size,
        shuffle=False,
        collate_fn=full_dataloader.collate_fn,
        drop_last=False,
        num_workers=0
    )
    
    return train_dataloader, val_dataloader


def setup_output_directory(args):
    """Create output directory structure."""
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/vanilla_transformer_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    return output_dir


def create_trainer(args, model, train_dataloader, val_dataloader, optimizer, device, output_dir):
    """Create the appropriate trainer based on arguments."""
    
    # Create JSON logger if not disabled
    json_logger = None
    if not args.disable_json_logging:
        json_logger = create_json_logger_for_training(
            output_dir=output_dir,
            experiment_name=args.experiment_name,
            log_every_n_steps=args.json_log_steps
        )
        logger.info(f"JSON logging enabled: {json_logger.log_file}")
    
    if args.use_accelerate:
        # Use accelerate trainer
        from accelerate import Accelerator
        accelerator = Accelerator()
        
        # Prepare model, optimizer, dataloader with accelerate
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        if val_dataloader:
            val_dataloader = accelerator.prepare(val_dataloader)
        
        trainer = create_accelerate_trainer_with_json_logging(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            accelerator=accelerator,
            output_dir=output_dir,
            experiment_name=args.experiment_name,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
            json_log_every_n_steps=args.json_log_steps,
            val_dataloader=val_dataloader,
            validate_every=args.validate_every,
            validate_every_n_batches=args.validate_every_n_batches
        )
    else:
        # Use simple trainer
        trainer = SimpleTrainerWithJSON(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
            output_dir=output_dir,
            json_logger=json_logger,
            val_dataloader=val_dataloader,
            validate_every=args.validate_every,
            validate_every_n_batches=args.validate_every_n_batches
        )
    
    return trainer


def run_generation_examples(model, tokenizer, device, prompts, epoch, json_logger=None):
    """Run text generation examples and log them."""
    logger.info(f"Running text generation examples for epoch {epoch}...")
    
    for prompt in prompts:
        try:
            generated_text = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8,
                device=device
            )
            
            logger.info(f"Generated text for '{prompt}': {generated_text}")
            
            # Log to JSON if available
            if json_logger:
                json_logger.log_generation(epoch, prompt, generated_text, {
                    'max_new_tokens': 50,
                    'temperature': 0.8
                })
        except Exception as e:
            logger.warning(f"Generation failed for prompt '{prompt}': {e}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Vanilla Transformer training with enhanced logging")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create configuration
    config = create_vanilla_config(args)
    logger.info("Model configuration:")
    print_config(config)
    
    # Create tokenizer and model
    tokenizer = create_tokenizer()
    model = get_model(config, tokenizer).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=getattr(config, 'weight_decay', 0.01)
    )
    
    # Load and split data
    train_dataloader, val_dataloader = load_and_split_data(args, tokenizer)
    logger.info(f"Data loaded: {len(train_dataloader)} training batches")
    if val_dataloader:
        logger.info(f"Validation: {len(val_dataloader)} validation batches")
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    logger.info(f"Output directory: {output_dir}")
    
    # Create trainer
    trainer = create_trainer(args, model, train_dataloader, val_dataloader, 
                           optimizer, device, output_dir)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        import json
        json.dump(vars(config), f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Training loop with generation examples
    logger.info("Starting training...")
    results = trainer.train()
    
    # Run final generation examples
    if args.generation_prompts:
        run_generation_examples(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=args.generation_prompts,
            epoch=args.num_epochs,
            json_logger=trainer.json_logger if hasattr(trainer, 'json_logger') else None
        )
    
    # Print final results
    logger.info("Training completed!")
    logger.info(f"Final training loss: {results.get('final_loss', 'N/A'):.4f}")
    if 'final_val_loss' in results and results['final_val_loss'] != float('nan'):
        logger.info(f"Final validation loss: {results['final_val_loss']:.4f}")
    logger.info(f"Total training time: {results.get('training_time', 0):.2f}s")
    logger.info(f"Output directory: {output_dir}")
    
    if hasattr(trainer, 'json_logger') and trainer.json_logger:
        logger.info(f"JSON logs saved to: {trainer.json_logger.log_file}")


if __name__ == "__main__":
    main()