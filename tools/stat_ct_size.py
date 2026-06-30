#!/usr/bin/env python3
"""
Summarize CT NIfTI voxel shapes before and after MONAI Spacingd.

Post-spacing shape is estimated with the same formula as the training pipeline:
    new_size[i] = round(original_size[i] * original_spacing[i] / target_spacing[i])

Usage:
  python tools/stat_ct_shapes.py --input_dir /path/to/grayscale
  python tools/stat_ct_shapes.py --json dataset.json --data_root /path/to/data --split training
  python tools/stat_ct_shapes.py --input_dir /path/to/grayscale --target_spacing 1.0 1.0 1.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm


def collect_nifti_paths(input_dir: Path, recursive: bool) -> List[Path]:
    """Collect NIfTI file paths from a directory."""
    patterns = ("**/*.nii.gz", "**/*.nii") if recursive else ("*.nii.gz", "*.nii")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(input_dir.glob(pattern)))

    # Deduplicate when both .nii and .nii.gz match the same basename
    seen = set()
    unique_paths: List[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def collect_paths_from_json(
    json_path: Path,
    data_root: Path,
    split: str,
    key: str,
) -> List[Path]:
    """Collect file paths from a dataset.json split and image key."""
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in {json_path.name}")

    paths: List[Path] = []
    for item in dataset[split]:
        if key not in item:
            raise KeyError(f"Key '{key}' not found in dataset entry")
        paths.append(data_root / item[key])
    return paths


def read_nifti_meta(path: Path) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """Read shape and spacing from the NIfTI header without loading voxel data."""
    img = nib.load(str(path))
    shape = tuple(int(s) for s in img.shape[:3])
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return shape, spacing


def compute_shape_after_spacing(
    shape: Tuple[int, ...],
    spacing: Tuple[float, ...],
    target_spacing: Tuple[float, float, float],
) -> Tuple[int, ...]:
    """Estimate voxel grid size after Spacingd (consistent with MONAI Spacingd)."""
    orig_shape = np.asarray(shape, dtype=np.float64)
    orig_spacing = np.asarray(spacing, dtype=np.float64)
    target = np.asarray(target_spacing, dtype=np.float64)
    new_shape = np.round(orig_shape * orig_spacing / target).astype(int)
    return tuple(int(s) for s in new_shape)


def format_stats(values: np.ndarray) -> str:
    """Format min / max / mean for one statistic column."""
    return f"min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}"


def print_shape_stats(
    title: str,
    shape_arr: np.ndarray,
    axis_names: Sequence[str],
) -> None:
    """Print per-axis and total voxel-count statistics."""
    print(title)
    print("-" * 72)
    for i, name in enumerate(axis_names):
        print(f"  {name}: {format_stats(shape_arr[:, i])}")
    voxel_counts = np.prod(shape_arr, axis=1)
    print(f"  Voxel count (D*H*W): {format_stats(voxel_counts)}")
    print()


def summarize(
    paths: Sequence[Path],
    target_spacing: Tuple[float, float, float],
    axis_names: Sequence[str] = ("D", "H", "W"),
) -> None:
    """Print shape and spacing statistics before and after Spacingd."""
    if not paths:
        print("No NIfTI files found.")
        return

    shapes_before: List[Tuple[int, ...]] = []
    shapes_after: List[Tuple[int, ...]] = []
    spacings: List[Tuple[float, ...]] = []
    missing_count = 0
    failed_count = 0

    for path in tqdm(paths, desc="Reading NIfTI", dynamic_ncols=True):
        if not path.exists():
            missing_count += 1
            continue
        try:
            shape, spacing = read_nifti_meta(path)
            shapes_before.append(shape)
            spacings.append(spacing)
            shapes_after.append(compute_shape_after_spacing(shape, spacing, target_spacing))
        except Exception as exc:
            failed_count += 1
            print(f"[WARN] Failed to read {path.name}: {exc}", file=sys.stderr)

    if not shapes_before:
        print("No valid NIfTI files could be read.")
        if missing_count:
            print(f"Missing files: {missing_count}")
        if failed_count:
            print(f"Failed reads: {failed_count}")
        return

    before_arr = np.array(shapes_before, dtype=np.float64)
    after_arr = np.array(shapes_after, dtype=np.float64)
    spacing_arr = np.array(spacings, dtype=np.float64)

    ts = ", ".join(f"{s:.3f}" for s in target_spacing)
    print("=" * 72)
    print("CT Size Statistics (Before / After Spacingd)")
    print("=" * 72)
    print(f"Files scanned     : {len(paths)}")
    print(f"Files valid       : {len(shapes_before)}")
    if missing_count:
        print(f"Files missing     : {missing_count}")
    if failed_count:
        print(f"Files failed      : {failed_count}")
    print(f"Target spacing    : ({ts}) mm")
    print()

    print_shape_stats("Before Spacingd — original voxel shape", before_arr, axis_names)

    print_shape_stats(
        f"After Spacingd — voxel shape at target spacing ({ts}) mm",
        after_arr,
        axis_names,
    )

    print("Original spacing (mm)")
    print("-" * 72)
    for i, name in enumerate(axis_names):
        print(f"  {name}: {format_stats(spacing_arr[:, i])}")
    print()

    # Voxel-count ratio: after spacing / before spacing
    voxels_before = np.prod(before_arr, axis=1)
    voxels_after = np.prod(after_arr, axis=1)
    ratio = voxels_after / np.maximum(voxels_before, 1)
    print("Voxel count change (after / before)")
    print("-" * 72)
    print(f"  Ratio: {format_stats(ratio)}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CT NIfTI shapes and spacings before/after Spacingd"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing CT NIfTI files",
    )
    source.add_argument(
        "--json",
        type=str,
        help="Path to dataset.json (requires --data_root)",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Root directory for relative paths in dataset.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "validation", "testing"],
        help="Dataset split to use (default: training)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="grayscale",
        help="Image key in dataset.json entries (default: grayscale)",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("SX", "SY", "SZ"),
        help="Target spacing for Spacingd in mm (default: 1.0 1.0 1.0)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_spacing = tuple(args.target_spacing)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        paths = collect_nifti_paths(input_dir, recursive=args.recursive)
    else:
        if not args.data_root:
            raise ValueError("--json mode requires --data_root")
        paths = collect_paths_from_json(
            json_path=Path(args.json),
            data_root=Path(args.data_root),
            split=args.split,
            key=args.key,
        )

    summarize(paths, target_spacing=target_spacing)


if __name__ == "__main__":
    main()
