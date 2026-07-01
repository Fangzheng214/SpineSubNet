#!/usr/bin/env python3
"""
Crop lumbar vertebra regions from CT volumes using segmentation masks.

For each CT/mask pair, extract one cropped CT per lumbar level (L1–L5, or L6 for VerSe)
based on the mask bounding box in voxel space.

Example:
    python tools/crop_lumbar_ct.py \\
        --ct_dir /path/to/rawdata \\
        --mask_dir /path/to/derivatives \\
        --output_dir /path/to/output \\
        --dataset verse \\
        --margin 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

LUMBAR_LABELS_VERSE: Dict[int, str] = {
    20: "L1",
    21: "L2",
    22: "L3",
    23: "L4",
    24: "L5",
    25: "L6",
}

LUMBAR_LABELS_COLON: Dict[int, str] = {
    20: "L1",
    21: "L2",
    22: "L3",
    23: "L4",
    24: "L5",
}

CT_SUFFIXES = (".nii.gz", ".nii")
MASK_KEYWORDS = ("seg-vert_msk", "seg_vert_msk", "mask", "seg")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def is_ct_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("_ct.nii.gz") or name.endswith("_ct.nii")


def is_mask_file(path: Path) -> bool:
    name = path.name.lower()
    if not (name.endswith(".nii.gz") or name.endswith(".nii")):
        return False
    return any(keyword in name for keyword in MASK_KEYWORDS)


def subject_id_from_path(path: Path) -> str:
    """Use parent folder name when BIDS-style, otherwise filename prefix."""
    if path.parent.name.startswith("sub-"):
        return path.parent.name
    stem = path.name
    for suffix in CT_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith("_ct"):
        stem = stem[: -len("_ct")]
    return stem


def collect_ct_files(ct_dir: Path) -> List[Path]:
    files = [p for p in ct_dir.rglob("*") if p.is_file() and is_ct_file(p)]
    return sorted(files)


def build_mask_index(mask_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for mask_path in mask_dir.rglob("*"):
        if not mask_path.is_file() or not is_mask_file(mask_path):
            continue
        subject = subject_id_from_path(mask_path)
        index.setdefault(subject, []).append(mask_path)
    for subject in index:
        index[subject] = sorted(index[subject])
    return index


def match_mask(ct_path: Path, mask_index: Dict[str, List[Path]]) -> Optional[Path]:
    subject = subject_id_from_path(ct_path)
    candidates = mask_index.get(subject, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    ct_stem = ct_path.name
    for suffix in CT_SUFFIXES:
        if ct_stem.endswith(suffix):
            ct_stem = ct_stem[: -len(suffix)]
            break

    for mask_path in candidates:
        mask_stem = mask_path.name
        for suffix in CT_SUFFIXES:
            if mask_stem.endswith(suffix):
                mask_stem = mask_stem[: -len(suffix)]
                break
        if ct_stem.replace("_ct", "") in mask_stem:
            return mask_path

    logging.warning(
        "Multiple masks for %s; using first match: %s",
        subject,
        candidates[0].name,
    )
    return candidates[0]


def lumbar_labels(dataset: str) -> Dict[int, str]:
    if dataset == "verse":
        return LUMBAR_LABELS_VERSE
    if dataset == "colon":
        return LUMBAR_LABELS_COLON
    raise ValueError(f"Unknown dataset type: {dataset}")


def bbox_slices(
    mask_data: np.ndarray,
    label_value: int,
    margin: int,
) -> Optional[Tuple[slice, slice, slice]]:
    coords = np.where(mask_data == label_value)
    if coords[0].size == 0:
        return None

    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) for c in coords]
    shape = mask_data.shape

    slices = tuple(
        slice(max(0, lo - margin), min(dim, hi + margin + 1))
        for lo, hi, dim in zip(mins, maxs, shape)
    )
    return slices


def crop_nifti(nii_img: nib.Nifti1Image, slices: Tuple[slice, slice, slice]) -> nib.Nifti1Image:
    data = np.asanyarray(nii_img.dataobj)[slices]
    start_ijk = [s.start for s in slices]
    new_affine = nii_img.affine.copy()
    new_affine[:3, 3] = nib.affines.apply_affine(nii_img.affine, start_ijk)

    header = nii_img.header.copy()
    header.set_data_shape(data.shape)
    return nib.Nifti1Image(data, new_affine, header)


def output_filename(ct_path: Path, vertebra_name: str) -> str:
    name = ct_path.name
    if name.endswith(".nii.gz"):
        return f"{name[:-7]}_{vertebra_name}.nii.gz"
    if name.endswith(".nii"):
        return f"{name[:-4]}_{vertebra_name}.nii"
    raise ValueError(f"Unsupported CT filename: {name}")


def process_case(
    ct_path: Path,
    mask_path: Path,
    output_dir: Path,
    labels: Dict[int, str],
    margin: int,
    preserve_structure: bool,
) -> int:
    ct_nii = nib.load(str(ct_path))
    mask_nii = nib.load(str(mask_path))

    ct_data = np.asanyarray(ct_nii.dataobj)
    mask_data = np.rint(np.asanyarray(mask_nii.dataobj)).astype(np.int16)

    if ct_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch for {ct_path.name}: CT {ct_data.shape} vs mask {mask_data.shape}"
        )

    saved = 0
    for label_value, vertebra_name in labels.items():
        slices = bbox_slices(mask_data, label_value, margin)
        if slices is None:
            logging.debug("%s: label %s (%s) not found, skip", ct_path.name, label_value, vertebra_name)
            continue

        cropped = crop_nifti(ct_nii, slices)
        out_name = output_filename(ct_path, vertebra_name)

        if preserve_structure:
            rel_parent = ct_path.parent.name if ct_path.parent.name.startswith("sub-") else ""
            out_path = output_dir / rel_parent / out_name if rel_parent else output_dir / out_name
        else:
            out_path = output_dir / out_name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(cropped, str(out_path))
        saved += 1
        logging.info("Saved %s", out_path)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop lumbar vertebra regions from CT using segmentation masks.",
    )
    parser.add_argument(
        "--ct_dir",
        type=Path,
        required=True,
        help="Directory containing CT NIfTI files (rawdata).",
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        required=True,
        help="Directory containing mask NIfTI files (derivatives).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save cropped CT volumes.",
    )
    parser.add_argument(
        "--dataset",
        type=str.strip,
        choices=["verse", "colon"],
        default="verse",
        help="Label scheme: verse=L1-L6 (20-25), colon=L1-L5 (20-24).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Extra voxels around each vertebra bounding box (default: 0).",
    )
    parser.add_argument(
        "--preserve_structure",
        action="store_true",
        help="Keep sub-XXX subfolders in output_dir.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    ct_dir = args.ct_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir
    labels = lumbar_labels(args.dataset)

    if not ct_dir.is_dir():
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ct_files = collect_ct_files(ct_dir)
    mask_index = build_mask_index(mask_dir)

    logging.info("Found %d CT files in %s", len(ct_files), ct_dir)
    logging.info("Lumbar labels: %s", ", ".join(f"{k}->{v}" for k, v in labels.items()))

    total_saved = 0
    skipped = 0
    failed = 0

    for ct_path in ct_files:
        mask_path = match_mask(ct_path, mask_index)
        if mask_path is None:
            logging.warning("No mask found for %s, skip", ct_path)
            skipped += 1
            continue

        try:
            saved = process_case(
                ct_path=ct_path,
                mask_path=mask_path,
                output_dir=output_dir,
                labels=labels,
                margin=args.margin,
                preserve_structure=args.preserve_structure,
            )
            total_saved += saved
            if saved == 0:
                logging.warning("No lumbar vertebra found in %s", mask_path.name)
        except Exception as exc:
            failed += 1
            logging.error("Failed %s: %s", ct_path.name, exc)

    logging.info(
        "Done. CT cases=%d, cropped volumes saved=%d, skipped=%d, failed=%d",
        len(ct_files),
        total_saved,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
