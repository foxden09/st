#./inference/__init__.py
"""
Inference module
Provides utilities for text generation and model inference
"""

from .generation import (
    run_generation,
    batch_generate
)

# export main functions
__all__ = [
    'run_generation',
    'batch_generate',
    'get_generation_args'
]
