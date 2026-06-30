"""
U-Net model for 3D segmentation
"""

from typing import Tuple, Optional
import logging

import torch
import torch.nn as nn
from monai.networks.nets import UNet


def create_unet_model(
    model_config: dict,
    img_size: Tuple[int, int, int],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create 3D U-Net segmentation model.

    Args:
        model_config: Model hyperparameters (in_channels, out_channels, etc.)
        img_size: Input image size (depth, height, width)
        device: Computing device (cuda/cpu). Auto-detected if None

    Returns:
        Initialized U-Net model on specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_dims = model_config.get('spatial_dims', 3)
    in_channels = model_config.get('in_channels', 1)
    out_channels = model_config.get('out_channels', 8)
    channels = model_config.get('channels', [16, 32, 64, 128, 256])
    strides = model_config.get('strides', [2, 2, 2, 2])
    num_res_units = model_config.get('num_res_units', 2)
    norm = model_config.get('norm_name', 'instance')
    dropout = model_config.get('dropout_rate', 0.0)

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    ).to(device)

    logging.info("U-Net Model Configuration:")
    logging.info(f"  - Spatial dimensions: {spatial_dims}D")
    logging.info(f"  - Input channels: {in_channels}")
    logging.info(f"  - Output classes: {out_channels}")
    logging.info(f"  - Channel sequence: {channels}")
    logging.info(f"  - Stride sequence: {strides}")
    logging.info(f"  - Residual units: {num_res_units}")
    logging.info(f"  - Normalization: {norm}")

    return model
