"""
Training script for grayscale baseline segmentation.

Usage:
    python train.py --config configs/baseline_grayscale.yaml
"""

import os
import logging

import torch
from monai.config import print_config
from monai.optimizers.lr_scheduler import WarmupCosineSchedule

from models import create_model
from trainers.baseline import BaselineTrainer
from data import get_baseline_transforms, create_data_loaders
from utils import (
    parse_arguments,
    setup_experiment_directory,
    set_random_seed,
)

print_config()


def create_optimizer_and_scheduler(config, model):
    """Create optimizer and optionally a scheduler."""
    optimizer_type = config.get('optimizer_type', 'AdamW')
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

    logging.info(f"Optimizer: {optimizer_type} (lr={config['lr']}, weight_decay={config['weight_decay']})")

    scheduler = None
    if 'scheduler' in config and config['scheduler'].get('use_scheduler', False):
        scheduler_config = config['scheduler']
        warmup_steps = scheduler_config.get('warmup_steps', 500)
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            t_total=config['max_iterations'],
            end_lr=scheduler_config.get('end_lr', 0.0),
            cycles=scheduler_config.get('cycles', 0.5),
            warmup_multiplier=scheduler_config.get('warmup_multiplier', 0.0),
        )
        logging.info(f"LR Scheduler: WarmupCosineSchedule (warmup_steps={warmup_steps})")
    else:
        logging.info("LR Scheduler: Not enabled")

    return optimizer, scheduler


def train_baseline(config, args):
    """Train baseline model on grayscale input."""
    logging.info("=" * 70)
    logging.info(" Baseline Model Training (Grayscale)")
    logging.info("=" * 70)

    logging.info("\nConfiguration Summary:")
    logging.info(f"  Data directory: {config['data_dir']}")
    logging.info(f"  Output directory: {config['root_dir']}")
    logging.info(f"  Batch size: {config['batch_size']}")
    logging.info(f"  Learning rate: {config['lr']}")
    logging.info(f"  Max iterations: {config['max_iterations']}")
    logging.info(f"  Random seed: {config['seed']}")

    set_random_seed(config['seed'])

    experiment_name = config.get('experiment_name', 'baseline_grayscale')
    writer, root_dir = setup_experiment_directory(
        config['root_dir'],
        experiment_name,
        args.config
    ) if config.get('use_tensorboard', True) else (None, config['root_dir'])

    logging.info("\n" + "=" * 70)
    logging.info(" Preparing Dataset and DataLoaders")
    logging.info("=" * 70)

    train_transforms, val_transforms = get_baseline_transforms(config)
    train_loader, val_loader = create_data_loaders(
        data_dir=config['data_dir'],
        json_file=config['json_file'],
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=config['batch_size'],
        val_batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        val_num_workers=config['val_num_workers'],
    )

    logging.info("\n" + "=" * 70)
    logging.info(" Creating Baseline Model")
    logging.info("=" * 70)

    model_config = config['model'].copy()
    model_config['in_channels'] = 1
    model = create_model(
        model_config=model_config,
        img_size=config['spatial_size'],
    )

    logging.info("\n" + "=" * 70)
    logging.info("Initializing Optimizer and Scheduler")
    logging.info("=" * 70)

    optimizer, scheduler = create_optimizer_and_scheduler(config, model)

    logging.info("\n" + "=" * 70)
    logging.info("Initializing Baseline Trainer")
    logging.info("=" * 70)

    if 'device' in config and config['device']:
        device = config['device']
        logging.info(f"Using device from config: {device}")
    elif os.environ.get('FORCE_CPU', '0') == '1':
        device = "cpu"
        logging.info("Using CPU (forced by FORCE_CPU environment variable)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Auto-detected device: {device}")

    trainer = BaselineTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        root_dir=root_dir,
        device=device,
    )

    trainer.train()


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    args, config = parse_arguments()

    trainer_type = config.get('trainer_type', 'baseline').lower()
    if trainer_type != 'baseline':
        raise ValueError(
            f"Unsupported trainer_type: {trainer_type}. Only 'baseline' is supported."
        )

    train_baseline(config, args)


if __name__ == "__main__":
    main()
