#!/usr/bin/env python3
"""
Full dataset generation script.

Combines the roles of generate_dataset_json.py and split_dataset.py:
writes a single JSON with training / validation / testing splits.
"""

import os
import json
import re
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import logging

# Match lumbar level suffix: *_L1.nii.gz, *_L5.nii, etc.
LEVEL_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.+)_L(\d+)(?:\.nii(?:\.gz)?)$", re.IGNORECASE)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def collect_valid_samples(data_dir: str) -> List[str]:
    """
    Collect filenames for all valid samples (binary, grayscale, label present).
    
    Args:
        data_dir: Root data directory with binary/, grayscale/, label/ subfolders.
    
    Returns:
        List of valid sample filenames.
    """
    binary_dir = os.path.join(data_dir, "binary")
    grayscale_dir = os.path.join(data_dir, "grayscale")
    label_dir = os.path.join(data_dir, "label")
    
    # Ensure required subdirectories exist
    for dir_path, dir_name in [(binary_dir, "binary"), (grayscale_dir, "grayscale"), (label_dir, "label")]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_name.capitalize()} directory not found: {dir_path}")
    
    # List files from binary/ (used as the master list)
    binary_files = sorted([f for f in os.listdir(binary_dir) if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    logging.info(f"Found {len(binary_files)} files in binary directory")
    
    # Keep only samples that have matching files in all three folders
    valid_samples = []
    missing_count = 0
    
    for filename in binary_files:
        binary_path = os.path.join(binary_dir, filename)
        grayscale_path = os.path.join(grayscale_dir, filename)
        label_path = os.path.join(label_dir, filename)
        
        if os.path.exists(binary_path) and os.path.exists(grayscale_path) and os.path.exists(label_path):
            valid_samples.append(filename)
        else:
            logging.warning(f"Missing files for {filename}")
            missing_count += 1
    
    logging.info(f"Valid samples: {len(valid_samples)}")
    if missing_count > 0:
        logging.warning(f"Skipped {missing_count} samples due to missing files")
    
    return valid_samples


def get_subject_prefix(filename: str) -> str | None:
    """
    Extract subject key from filename by stripping the lumbar level suffix.

    Examples:
        sub-verse010_ct_L1.nii.gz       -> sub-verse010_ct
        sub-verse521_dir-ax_ct_L5.nii.gz -> sub-verse521_dir-ax_ct
    """
    match = LEVEL_SUFFIX_PATTERN.match(filename)
    if match is None:
        return None
    return match.group("prefix")


def group_samples_by_subject(samples: List[str]) -> Dict[str, List[str]]:
    """
    Group sample filenames by subject (common prefix before _L{level}).

    Each subject may contain any subset of lumbar levels (L1-L5, or fewer).
    Files that do not match the level suffix pattern are treated as singleton
    subjects using the full filename as the key.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    unmatched: List[str] = []

    for filename in samples:
        prefix = get_subject_prefix(filename)
        if prefix is None:
            unmatched.append(filename)
            groups[filename].append(filename)
        else:
            groups[prefix].append(filename)

    for filename in sorted(groups):
        groups[filename].sort()

    if unmatched:
        logging.warning(
            "Could not parse lumbar level suffix for %d file(s); "
            "each is treated as its own subject group: %s",
            len(unmatched),
            ", ".join(unmatched[:5]) + (" ..." if len(unmatched) > 5 else ""),
        )

    return dict(groups)


def split_subjects(
    subject_groups: Dict[str, List[str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split subjects into train / val / test by ratio, then expand to all levels.

    Splitting is done at subject level so all lumbar levels from the same
    person stay in the same split.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    random.seed(seed)

    subject_keys = list(subject_groups.keys())
    random.shuffle(subject_keys)

    total_subjects = len(subject_keys)
    train_end = int(total_subjects * train_ratio)
    val_end = train_end + int(total_subjects * val_ratio)

    train_subjects = subject_keys[:train_end]
    val_subjects = subject_keys[train_end:val_end]
    test_subjects = subject_keys[val_end:]

    def expand(subjects: List[str]) -> List[str]:
        files: List[str] = []
        for subject in subjects:
            files.extend(subject_groups[subject])
        return sorted(files)

    train_samples = expand(train_subjects)
    val_samples = expand(val_subjects)
    test_samples = expand(test_subjects)

    total_samples = len(train_samples) + len(val_samples) + len(test_samples)
    levels_per_subject = [len(subject_groups[k]) for k in subject_keys]

    logging.info("Subject groups: %d (samples: %d)", total_subjects, total_samples)
    logging.info(
        "  Levels per subject: min=%d, max=%d, mean=%.1f",
        min(levels_per_subject),
        max(levels_per_subject),
        sum(levels_per_subject) / total_subjects,
    )
    logging.info("Subject split:")
    logging.info(
        "  Training:   %4d subjects (%5.1f%%), %4d samples",
        len(train_subjects),
        len(train_subjects) / total_subjects * 100,
        len(train_samples),
    )
    logging.info(
        "  Validation: %4d subjects (%5.1f%%), %4d samples",
        len(val_subjects),
        len(val_subjects) / total_subjects * 100,
        len(val_samples),
    )
    logging.info(
        "  Testing:    %4d subjects (%5.1f%%), %4d samples",
        len(test_subjects),
        len(test_subjects) / total_subjects * 100,
        len(test_samples),
    )

    return {
        "training": train_samples,
        "validation": val_samples,
        "testing": test_samples,
    }


def split_samples(
    samples: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split samples by subject groups (all lumbar levels stay together)."""
    subject_groups = group_samples_by_subject(samples)
    return split_subjects(
        subject_groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def create_dataset_dict(
    split_samples: Dict[str, List[str]],
    use_relative_paths: bool = True
) -> Dict[str, Any]:
    """
    Build a MONAI-style dataset dictionary.
    
    Args:
        split_samples: Dict of splits (training / validation / testing filenames).
        use_relative_paths: If True, paths are relative to data_dir.
    
    Returns:
        Dataset dict in MONAI JSON convention.
    """
    def create_sample_entry(filename: str) -> Dict[str, str]:
        """Build one sample entry (paths under binary/grayscale/label)."""
        if use_relative_paths:
            return {
                "binary": os.path.join("binary", filename),
                "grayscale": os.path.join("grayscale", filename),
                "label": os.path.join("label", filename)
            }
        else:
            # Same layout; extend here if absolute paths are required
            return {
                "binary": os.path.join("binary", filename),
                "grayscale": os.path.join("grayscale", filename),
                "label": os.path.join("label", filename)
            }
    
    training_list = [create_sample_entry(f) for f in split_samples['training']]
    validation_list = [create_sample_entry(f) for f in split_samples['validation']]
    testing_list = [create_sample_entry(f) for f in split_samples['testing']]
    
    dataset_dict = {
        "name": "Vertebrae Subregion Segmentation",
        "description": "Vertebrae subregion segmentation dataset with binary and grayscale inputs",
        "reference": "Custom dataset",
        "licence": "N/A",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {
            "0": "CT"
        },
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
    
    return dataset_dict


def save_dataset_json(dataset_dict: Dict[str, Any], output_path: str):
    """
    Write the dataset dict to a JSON file.
    
    Args:
        dataset_dict: Dataset dictionary to serialize.
        output_path: Output file path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
    
    logging.info(f"✓ Dataset JSON file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate complete dataset JSON file with train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_complete_dataset.py --data_dir /root/autodl-tmp/subregion --output dataset.json

This will create a dataset.json file with subject-level splits:
  - Samples sharing the same prefix before _L{level} stay in the same split
  - Training set (70% of subjects)
  - Validation set (10% of subjects)
  - Testing set (20% of subjects)
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory containing binary/, grayscale/, label/ folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.json",
        help="Output JSON filename (default: dataset.json)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test set ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("=" * 70)
    logging.info("Complete Dataset Generation Tool")
    logging.info("=" * 70)
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    logging.info(f"Random seed: {args.seed}")
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
    dataset_dict = create_dataset_dict(split_dict, use_relative_paths=True)
    
    output_path = os.path.join(args.data_dir, args.output)
    save_dataset_json(dataset_dict, output_path)
    
    logging.info("\n" + "=" * 70)
    logging.info("Dataset Summary")
    logging.info("=" * 70)
    logging.info(f"Training samples:   {dataset_dict['numTraining']:4d}")
    logging.info(f"Validation samples: {dataset_dict['numValidation']:4d}")
    logging.info(f"Testing samples:    {dataset_dict['numTesting']:4d}")
    logging.info(f"Total samples:      {dataset_dict['numTraining'] + dataset_dict['numValidation'] + dataset_dict['numTesting']:4d}")
    logging.info("=" * 70)
    
    logging.info(f"\n✓ Done! You can now use this dataset in your training pipeline.")
    logging.info(f"\nTo use this dataset, update your config file:")
    logging.info(f"  data:")
    logging.info(f"    data_dir: \"{args.data_dir}\"")
    logging.info(f"    json_file: \"{args.output}\"")


if __name__ == "__main__":
    main()

