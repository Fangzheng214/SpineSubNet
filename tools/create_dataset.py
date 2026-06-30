#!/usr/bin/env python3
"""
Dataset JSON generation script.

Creates a MONAI Decathlon-style dataset.json with training / validation / testing splits
from grayscale/ and label/ subdirectories.
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any
import logging


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def collect_valid_samples(data_dir: str) -> List[str]:
    """
    Collect filenames for all valid samples (grayscale + label present).

    Args:
        data_dir: Root data directory with grayscale/ and label/ subfolders.

    Returns:
        List of valid sample filenames.
    """
    grayscale_dir = os.path.join(data_dir, "grayscale")
    label_dir = os.path.join(data_dir, "label")

    for dir_path, dir_name in [(grayscale_dir, "grayscale"), (label_dir, "label")]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_name.capitalize()} directory not found: {dir_path}")

    grayscale_files = sorted([
        f for f in os.listdir(grayscale_dir)
        if f.endswith('.nii.gz') or f.endswith('.nii')
    ])

    logging.info(f"Found {len(grayscale_files)} files in grayscale directory")

    valid_samples = []
    missing_count = 0

    for filename in grayscale_files:
        grayscale_path = os.path.join(grayscale_dir, filename)
        label_path = os.path.join(label_dir, filename)

        if os.path.exists(grayscale_path) and os.path.exists(label_path):
            valid_samples.append(filename)
        else:
            logging.warning(f"Missing label for {filename}")
            missing_count += 1

    logging.info(f"Valid samples: {len(valid_samples)}")
    if missing_count > 0:
        logging.warning(f"Skipped {missing_count} samples due to missing files")

    return valid_samples


def split_samples(
    samples: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, List[str]]:
    """Split sample filenames into train / val / test by ratio."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    random.seed(seed)

    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)

    total = len(shuffled_samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_samples = shuffled_samples[:train_end]
    val_samples = shuffled_samples[train_end:val_end]
    test_samples = shuffled_samples[val_end:]

    logging.info("Dataset split:")
    logging.info(f"  Training:   {len(train_samples):4d} samples ({len(train_samples)/total*100:.1f}%)")
    logging.info(f"  Validation: {len(val_samples):4d} samples ({len(val_samples)/total*100:.1f}%)")
    logging.info(f"  Testing:    {len(test_samples):4d} samples ({len(test_samples)/total*100:.1f}%)")

    return {
        'training': train_samples,
        'validation': val_samples,
        'testing': test_samples
    }


def create_dataset_dict(split_samples: Dict[str, List[str]]) -> Dict[str, Any]:
    """Build a MONAI-style dataset dictionary."""

    def create_sample_entry(filename: str) -> Dict[str, str]:
        return {
            "grayscale": os.path.join("grayscale", filename),
            "label": os.path.join("label", filename),
        }

    training_list = [create_sample_entry(f) for f in split_samples['training']]
    validation_list = [create_sample_entry(f) for f in split_samples['validation']]
    testing_list = [create_sample_entry(f) for f in split_samples['testing']]

    return {
        "name": "Vertebrae Subregion Segmentation",
        "description": "Vertebrae subregion segmentation dataset with grayscale CT input",
        "reference": "Custom dataset",
        "licence": "N/A",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {
            "0": "background",
            "1": "region_1",
            "2": "region_2",
            "3": "region_3",
            "4": "region_4",
            "5": "region_5",
            "6": "region_6",
            "7": "region_7"
        },
        "numTraining": len(training_list),
        "numValidation": len(validation_list),
        "numTesting": len(testing_list),
        "training": training_list,
        "validation": validation_list,
        "testing": testing_list
    }


def save_dataset_json(dataset_dict: Dict[str, Any], output_path: str):
    """Write the dataset dict to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

    logging.info(f"Dataset JSON saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset JSON file with train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python tools/create_dataset.py --data_dir /path/to/data --output dataset.json
        """
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory containing grayscale/ and label/ folders")
    parser.add_argument("--output", type=str, default="dataset.json",
                        help="Output JSON filename (default: dataset.json)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Training set ratio (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    setup_logging()

    logging.info("=" * 70)
    logging.info("Dataset Generation Tool")
    logging.info("=" * 70)
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    logging.info("=" * 70)

    logging.info("\n[1/3] Collecting valid samples...")
    valid_samples = collect_valid_samples(args.data_dir)

    if len(valid_samples) == 0:
        logging.error("No valid samples found! Exiting.")
        return

    logging.info("\n[2/3] Splitting dataset...")
    split_dict = split_samples(
        valid_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    logging.info("\n[3/3] Creating dataset dictionary...")
    dataset_dict = create_dataset_dict(split_dict)

    output_path = os.path.join(args.data_dir, args.output)
    save_dataset_json(dataset_dict, output_path)

    total = dataset_dict['numTraining'] + dataset_dict['numValidation'] + dataset_dict['numTesting']
    logging.info("\n" + "=" * 70)
    logging.info("Dataset Summary")
    logging.info("=" * 70)
    logging.info(f"Training samples:   {dataset_dict['numTraining']:4d}")
    logging.info(f"Validation samples: {dataset_dict['numValidation']:4d}")
    logging.info(f"Testing samples:    {dataset_dict['numTesting']:4d}")
    logging.info(f"Total samples:      {total:4d}")
    logging.info("=" * 70)

    logging.info(f"\nUpdate your config:")
    logging.info(f"  data:")
    logging.info(f"    data_dir: \"{args.data_dir}\"")
    logging.info(f"    json_file: \"{args.output}\"")


if __name__ == "__main__":
    main()
