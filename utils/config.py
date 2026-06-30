"""
Configuration loading and parsing utilities
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logging.info(f"✓ Loaded config file: {config_path}")
    return config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML configuration with command-line arguments."""
    flat_config = {}

    if 'trainer_type' in config:
        flat_config['trainer_type'] = config['trainer_type']
    if 'device' in config:
        flat_config['device'] = config['device']

    if 'model' in config:
        flat_config['model'] = config['model']

    if 'data' in config:
        flat_config['data_dir'] = config['data'].get('data_dir', '')
        flat_config['json_file'] = config['data'].get('json_file', '')
        flat_config['spatial_size'] = tuple(config['data'].get('spatial_size', [160, 160, 160]))
        flat_config['target_spacing'] = tuple(config['data'].get('target_spacing', [1.0, 1.0, 1.0]))
        flat_config['use_spacing'] = config['data'].get('use_spacing', True)

    if 'training' in config:
        flat_config['max_iterations'] = config['training'].get('max_iterations', 5000)
        flat_config['eval_num'] = config['training'].get('eval_num', 50)
        flat_config['batch_size'] = config['training'].get('batch_size', 2)
        flat_config['val_batch_size'] = config['training'].get('val_batch_size', 1)
        flat_config['seed'] = config['training'].get('seed', 42)

    if 'augmentation' in config:
        flat_config['augmentation'] = config['augmentation']

    if 'optimizer' in config:
        flat_config['lr'] = config['optimizer'].get('lr', 1e-4)
        flat_config['weight_decay'] = config['optimizer'].get('weight_decay', 1e-5)
        flat_config['optimizer_type'] = config['optimizer'].get('type', 'AdamW')

    if 'scheduler' in config:
        flat_config['scheduler'] = config['scheduler']

    if 'dataloader' in config:
        flat_config['num_workers'] = config['dataloader'].get('num_workers', 8)
        flat_config['val_num_workers'] = config['dataloader'].get('val_num_workers', 4)
        flat_config['pin_memory'] = config['dataloader'].get('pin_memory', True)

    if 'experiment' in config:
        flat_config['root_dir'] = config['experiment'].get('root_dir', 'experiments')
        flat_config['experiment_name'] = config['experiment'].get('name', 'baseline_grayscale')
        flat_config['use_tensorboard'] = config['experiment'].get('use_tensorboard', True)

    if hasattr(args, 'data_dir') and args.data_dir is not None:
        flat_config['data_dir'] = args.data_dir
    if hasattr(args, 'root_dir') and args.root_dir is not None:
        flat_config['root_dir'] = args.root_dir
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        flat_config['batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr is not None:
        flat_config['lr'] = args.lr

    return flat_config


def parse_arguments() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse command-line arguments and load configuration."""
    parser = argparse.ArgumentParser(
        description="Grayscale Baseline Segmentation Training\n"
                    "Supports YAML config and command-line overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--root_dir", type=str, default=None, help="Experiment output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")

    args = parser.parse_args()

    yaml_config = load_config(args.config)
    final_config = merge_config_with_args(yaml_config, args)

    return args, final_config
