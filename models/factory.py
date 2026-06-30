"""
Model factory for creating segmentation models
"""

import os
import logging
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from .unet import create_unet_model


def create_model(
    model_config: Dict[str, Any],
    img_size: Tuple[int, int, int],
) -> nn.Module:
    """
    Factory function to create segmentation model based on configuration.

    Args:
        model_config: Model configuration dictionary
        img_size: Input image size (depth, height, width)

    Returns:
        Initialized PyTorch model on available device
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model_type = model_config.get('type', 'unet').lower()
    logging.info(f"\nCreating model type: {model_type.upper()}")

    if model_type != 'unet':
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: 'unet'"
        )

    model = create_unet_model(model_config, img_size, device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("\nModel Statistics:")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params:,}")

    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Extract model information for logging and analysis."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_type': model.__class__.__name__,
    }
