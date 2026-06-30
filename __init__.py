"""
Vertebra Subregion Segmentation Framework

A modular deep learning framework for 3D vertebra subregion segmentation
using grayscale CT volumes.
"""

__version__ = "1.0.0"

from .models import create_model, get_model_info
from .trainers import BaselineTrainer
from .data import get_baseline_transforms, create_data_loaders
from .losses import CombinedCEDiceLoss
from .utils import load_config, setup_logging, set_random_seed

__all__ = [
    'create_model',
    'get_model_info',
    'BaselineTrainer',
    'get_baseline_transforms',
    'create_data_loaders',
    'CombinedCEDiceLoss',
    'load_config',
    'setup_logging',
    'set_random_seed',
]
