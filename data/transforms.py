"""
Data transforms for training and validation
"""

import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CastToTyped,
    Orientationd,
    Spacingd,
    EnsureTyped,
    SpatialPadd,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
)


def _build_base_transforms(
    keys: List[str],
    spatial_size: Tuple[int, int, int],
    target_spacing: Tuple[float, float, float],
    use_spacing: bool,
) -> List:
    """Build base transforms (load, spacing, normalization, padding)."""
    data_keys = [k for k in keys if k != "label"]

    transform_list = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        CastToTyped(keys=data_keys, dtype=np.float32),
    ]

    if "label" in keys:
        transform_list.append(CastToTyped(keys=["label"], dtype=np.int16))

    if use_spacing:
        mode_list = ["nearest" if key == "label" else "bilinear" for key in keys]
        transform_list.extend([
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=target_spacing, mode=mode_list),
        ])

    transform_list.append(
        ScaleIntensityRanged(
            keys=["grayscale"],
            a_min=-500,
            a_max=1500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    )

    transform_list.append(EnsureTyped(keys=data_keys, dtype=torch.float32))
    if "label" in keys:
        transform_list.append(EnsureTyped(keys=["label"], dtype=torch.int16))
    transform_list.append(
        SpatialPadd(
            keys=keys,
            spatial_size=spatial_size,
            mode="constant",
            constant_values=0,
        )
    )

    return transform_list


def _build_augmentation_transforms(
    keys: List[str],
    flip_prob: float,
    rotation_prob: float,
    rotation_max_k: int,
) -> List:
    """Build data augmentation transforms."""
    return [
        RandFlipd(keys=keys, spatial_axis=[0], prob=flip_prob),
        RandFlipd(keys=keys, spatial_axis=[1], prob=flip_prob),
        RandFlipd(keys=keys, spatial_axis=[2], prob=flip_prob),
        RandRotate90d(
            keys=keys,
            prob=rotation_prob,
            max_k=rotation_max_k,
            spatial_axes=(0, 1)
        ),
    ]


def get_baseline_transforms(config: Dict[str, Any]) -> Tuple[Compose, Compose]:
    """
    Create transforms for baseline grayscale training.

    Args:
        config: Configuration dictionary

    Returns:
        train_transforms, val_transforms
    """
    spatial_size = config['spatial_size']
    target_spacing = config['target_spacing']
    use_spacing = config.get('use_spacing', True)
    keys = ["grayscale", "label"]

    aug_config = config.get('augmentation', {})
    flip_prob = aug_config.get('flip_prob', 0.10)
    rotation_prob = aug_config.get('rotation_prob', 0.10)
    rotation_max_k = aug_config.get('rotation_max_k', 3)

    base_transforms = _build_base_transforms(
        keys, spatial_size, target_spacing, use_spacing
    )
    aug_transforms = _build_augmentation_transforms(
        keys, flip_prob, rotation_prob, rotation_max_k
    )

    train_transforms = Compose(base_transforms + aug_transforms)
    val_transforms = Compose(base_transforms)

    logging.info("Baseline transforms created (grayscale + label)")
    return train_transforms, val_transforms
