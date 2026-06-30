"""
Data loader creation utilities
"""

import os
import logging
from typing import Tuple

from torch.utils.data import DataLoader
from monai.data import Dataset, load_decathlon_datalist
from monai.transforms import Compose


def create_data_loaders(
    data_dir: str,
    json_file: str,
    train_transforms: Compose,
    val_transforms: Compose,
    batch_size: int = 2,
    val_batch_size: int = 1,
    num_workers: int = 8,
    val_num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_dir: Root directory containing data
        json_file: JSON file with data list (relative to data_dir)
        train_transforms: Training transforms
        val_transforms: Validation transforms
        batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: Number of workers for training loader
        val_num_workers: Number of workers for validation loader
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    dataset_json_path = os.path.join(data_dir, json_file)
    logging.info(f"Using dataset JSON file: {dataset_json_path}")
    
    # Load data lists
    train_datalist = load_decathlon_datalist(dataset_json_path, True, "training")
    val_datalist = load_decathlon_datalist(dataset_json_path, True, "validation")
    
    logging.info(f"Number of training samples: {len(train_datalist)}")
    logging.info(f"Number of validation samples: {len(val_datalist)}")
    
    # Create datasets
    train_dataset = Dataset(data=train_datalist, transform=train_transforms)
    val_dataset = Dataset(data=val_datalist, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
    )
    
    logging.info(f"Train loader: {len(train_loader)} batches")
    logging.info(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader

