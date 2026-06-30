#!/usr/bin/env python3
"""
Model evaluation script — evaluates a trained baseline model on a test set.

Metrics:
  - Dice coefficient (DSC)
  - Hausdorff distance 95% (HD95)

Usage:
  python tools/evaluate_model.py \
    --model_path experiments/baseline_grayscale/weights/best_metric.pth \
    --config configs/baseline_grayscale.yaml \
    --test_json /path/to/dataset.json \
    --output_dir evaluation_results/baseline
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from monai.config import print_config
from monai.data import DataLoader, Dataset, load_decathlon_datalist, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose

sys.path.append(str(Path(__file__).parent.parent))
from models import create_model
from data.transforms import get_baseline_transforms

print_config()


def setup_logging(output_dir: str):
    """Configure logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"evaluation_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info(f"Log file: {log_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested training config for transform building."""
    flat = {}
    if 'data' in config:
        flat['spatial_size'] = tuple(config['data'].get('spatial_size', [160, 160, 160]))
        flat['target_spacing'] = tuple(config['data'].get('target_spacing', [1.0, 1.0, 1.0]))
        flat['use_spacing'] = config['data'].get('use_spacing', True)
        flat['data_dir'] = config['data'].get('data_dir', '')
    if 'augmentation' in config:
        flat['augmentation'] = config['augmentation']
    return flat


def load_model(model_path: str, config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Load a trained checkpoint into the baseline U-Net."""
    model_config = config.get('model', {}).copy()
    model_config['in_channels'] = 1

    model = create_model(
        model_config=model_config,
        img_size=config['data']['spatial_size']
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[5:] if key.startswith('unet.') else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    logging.info(f"Model loaded from: {model_path}")
    logging.info(f"  Input channels: 1 (grayscale)")

    return model


def create_test_dataloader(
    test_json: str,
    config: Dict[str, Any],
    test_transforms: Compose,
    batch_size: int = 1
) -> Tuple[DataLoader, List[Dict[str, Any]]]:
    """Build test DataLoader from Decathlon-style JSON."""
    data_dir = config.get("data", {}).get("data_dir")
    base_dir = Path(data_dir) if data_dir else None
    test_datalist = load_decathlon_datalist(
        test_json, True, "testing", base_dir=base_dir
    )

    logging.info(f"Number of test samples: {len(test_datalist)}")

    test_dataset = Dataset(data=test_datalist, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader, test_datalist


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 8,
    skip_hd95: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run inference and compute Dice / HD95 on the test loader."""
    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=False
    )

    if not skip_hd95:
        hd95_metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=95.0,
            reduction="mean_batch",
            get_not_nans=False
        )
        logging.info("HD95 metric enabled")
    else:
        hd95_metric = None
        logging.info("HD95 disabled (skip_hd95=True) to save memory")

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    model.eval()

    per_sample_results = []
    all_dice_scores = []
    all_hd95_scores = []

    with torch.no_grad():
        epoch_iterator = tqdm(test_loader, desc="Evaluating", dynamic_ncols=True)

        for batch_idx, batch in enumerate(epoch_iterator):
            labels = batch["label"].to(device)
            inputs = batch["grayscale"].to(device)

            outputs = model(inputs)

            labels_list = decollate_batch(labels)
            outputs_list = decollate_batch(outputs)

            labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
            outputs_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]

            dice_metric(y_pred=outputs_convert, y=labels_convert)
            batch_dice = dice_metric.aggregate()

            if hd95_metric is not None:
                hd95_metric(y_pred=outputs_convert, y=labels_convert)
                batch_hd95 = hd95_metric.aggregate()
            else:
                if isinstance(batch_dice, torch.Tensor):
                    batch_dice_np = batch_dice.cpu().numpy()
                    batch_hd95 = np.full_like(batch_dice_np, np.inf)
                    batch_dice = batch_dice_np
                else:
                    batch_hd95 = np.full_like(batch_dice, np.inf)

            if isinstance(batch_dice, torch.Tensor):
                batch_dice = batch_dice.cpu().numpy()
            if isinstance(batch_hd95, torch.Tensor):
                batch_hd95 = batch_hd95.cpu().numpy()

            all_dice_scores.append(batch_dice)
            all_hd95_scores.append(batch_hd95)

            if len(batch_dice.shape) == 1:
                per_sample_results.append({
                    'sample_idx': batch_idx,
                    'dice_per_class': batch_dice.tolist(),
                    'dice_mean': float(np.mean(batch_dice)),
                    'hd95_per_class': batch_hd95.tolist(),
                    'hd95_mean': float(np.mean(batch_hd95[np.isfinite(batch_hd95)])) if np.any(np.isfinite(batch_hd95)) else float('inf')
                })

            dice_metric.reset()
            if hd95_metric is not None:
                hd95_metric.reset()

            epoch_iterator.set_description(f"Evaluating (Dice={np.mean(batch_dice):.4f})")

    all_dice_scores = np.array(all_dice_scores)
    all_hd95_scores = np.array(all_hd95_scores)
    finite_hd95 = all_hd95_scores[np.isfinite(all_hd95_scores)]

    overall_stats = {
        'num_samples': len(all_dice_scores),
        'num_classes': num_classes,
        'dice_mean': float(np.mean(all_dice_scores)),
        'dice_std': float(np.std(all_dice_scores)),
        'dice_per_class_mean': np.mean(all_dice_scores, axis=0).tolist(),
        'dice_per_class_std': np.std(all_dice_scores, axis=0).tolist(),
        'hd95_mean': float(np.mean(finite_hd95)) if len(finite_hd95) > 0 else float('inf'),
        'hd95_std': float(np.std(finite_hd95)) if len(finite_hd95) > 0 else 0.0,
        'hd95_per_class_mean': np.nanmean(all_hd95_scores, axis=0).tolist(),
        'hd95_per_class_std': np.nanstd(all_hd95_scores, axis=0).tolist(),
    }

    return overall_stats, per_sample_results


def save_results(
    overall_stats: Dict[str, Any],
    per_sample_results: List[Dict[str, Any]],
    output_dir: str,
    test_datalist: List[Dict[str, Any]]
):
    """Write overall stats (JSON), per-sample table (CSV), and summary (TXT)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    stats_file = os.path.join(output_dir, f"overall_stats_{timestamp}.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2)
    logging.info(f"Overall statistics saved to: {stats_file}")

    if per_sample_results:
        for idx, result in enumerate(per_sample_results):
            if idx < len(test_datalist):
                result['grayscale_path'] = test_datalist[idx].get('grayscale', '')
                result['label_path'] = test_datalist[idx].get('label', '')

        df = pd.DataFrame(per_sample_results)
        csv_file = os.path.join(output_dir, f"per_sample_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        logging.info(f"Per-sample results saved to: {csv_file}")

    txt_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Model Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of samples: {overall_stats['num_samples']}\n")
        f.write(f"Number of classes: {overall_stats['num_classes']}\n\n")
        f.write("-" * 70 + "\n")
        f.write("Dice Coefficient (DSC)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Dice (all classes): {overall_stats['dice_mean']:.4f} ± {overall_stats['dice_std']:.4f}\n\n")
        f.write("Per-class Dice scores:\n")
        for idx, (mean, std) in enumerate(zip(overall_stats['dice_per_class_mean'], overall_stats['dice_per_class_std'])):
            f.write(f"  Class {idx}: {mean:.4f} ± {std:.4f}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("Hausdorff Distance 95% (HD95)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean HD95 (all classes): {overall_stats['hd95_mean']:.4f} ± {overall_stats['hd95_std']:.4f}\n\n")
        f.write("Per-class HD95 scores:\n")
        for idx, (mean, std) in enumerate(zip(overall_stats['hd95_per_class_mean'], overall_stats['hd95_per_class_std'])):
            f.write(f"  Class {idx}: {mean:.4f} ± {std:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")

    logging.info(f"Evaluation summary saved to: {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained baseline model on the test set",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained weights (.pth)")
    parser.add_argument("--config", type=str, required=True,
                        help="Training YAML config path")
    parser.add_argument("--test_json", type=str, required=True,
                        help="Dataset JSON with a 'testing' split")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory for evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size")
    parser.add_argument("--skip_hd95", action="store_true",
                        help="Skip HD95 to save GPU memory")

    args = parser.parse_args()

    setup_logging(args.output_dir)

    logging.info("=" * 70)
    logging.info("Baseline Model Evaluation")
    logging.info("=" * 70)
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Config: {args.config}")
    logging.info(f"Test JSON: {args.test_json}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\nDevice: {device}")

    logging.info("\n[1/5] Loading configuration...")
    config = load_config(args.config)

    logging.info("\n[2/5] Building transforms...")
    flat_config = flatten_config(config)
    _, test_transforms = get_baseline_transforms(flat_config)

    logging.info("\n[3/5] Loading test dataset...")
    test_loader, test_datalist = create_test_dataloader(
        args.test_json, config, test_transforms, batch_size=args.batch_size
    )

    logging.info("\n[4/5] Loading model...")
    model = load_model(args.model_path, config, device)

    logging.info("\n[5/5] Evaluating...")
    num_classes = config.get('model', {}).get('out_channels', 8)

    overall_stats, per_sample_results = evaluate_model(
        model, test_loader, device,
        num_classes=num_classes,
        skip_hd95=args.skip_hd95
    )

    logging.info("\n" + "=" * 70)
    logging.info("Results")
    logging.info("=" * 70)
    logging.info(f"Mean Dice: {overall_stats['dice_mean']:.4f} ± {overall_stats['dice_std']:.4f}")
    logging.info(f"Mean HD95: {overall_stats['hd95_mean']:.4f} ± {overall_stats['hd95_std']:.4f}")
    logging.info("=" * 70)

    logging.info("\nSaving results...")
    save_results(overall_stats, per_sample_results, args.output_dir, test_datalist)

    logging.info("\nEvaluation finished.")
    logging.info(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
