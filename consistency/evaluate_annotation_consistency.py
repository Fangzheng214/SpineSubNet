"""Evaluate inter-rater consistency for vertebra substructure annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import pstdev

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each doctor annotation against the original baseline labels. "
            "This script does not compute pairwise doctor-to-doctor metrics."
        )
    )
    parser.add_argument(
        "--baseline-label-dir",
        type=Path,
        required=True,
        help="Directory containing original baseline NIfTI label files.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory containing rater subdirectories, e.g. doctor_01 ... doctor_06.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV and JSON metric reports.",
    )
    parser.add_argument(
        "--raters",
        nargs="*",
        default=None,
        help="Optional rater directory names. Default: all immediate subdirectories.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include label 0 in per-label metrics. Default: false.",
    )
    parser.add_argument(
        "--check-foreground-union",
        action="store_true",
        help="Require each rater to have the same non-zero foreground mask as the baseline.",
    )
    return parser.parse_args()


def is_nifti_path(path: Path) -> bool:
    return path.name.endswith(NIFTI_SUFFIXES)


def strip_nifti_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def normalize_case_id(relative_path: Path, rater_name: str) -> str:
    stem = strip_nifti_suffix(relative_path.name)
    suffix = f"_{rater_name}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return str(relative_path.parent / stem).replace("\\", "/")


def find_rater_dirs(annotations_dir: Path, requested_raters: list[str] | None) -> list[Path]:
    if requested_raters:
        rater_dirs = [annotations_dir / rater_name for rater_name in requested_raters]
    else:
        rater_dirs = sorted(path for path in annotations_dir.iterdir() if path.is_dir())
    missing = [str(path) for path in rater_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing rater directories: {missing}")
    if len(rater_dirs) < 2:
        raise ValueError("At least two rater directories are required.")
    return rater_dirs


def collect_cases(rater_dirs: list[Path]) -> dict[str, dict[str, Path]]:
    case_map: dict[str, dict[str, Path]] = defaultdict(dict)
    for rater_dir in rater_dirs:
        rater_name = rater_dir.name
        for path in sorted(rater_dir.rglob("*")):
            if not path.is_file() or not is_nifti_path(path):
                continue
            relative_path = path.relative_to(rater_dir)
            case_id = normalize_case_id(relative_path, rater_name)
            if rater_name in case_map[case_id]:
                raise ValueError(f"Duplicate case for {rater_name}: {case_id}")
            case_map[case_id][rater_name] = path
    return dict(case_map)


def read_label(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    label_array = sitk.GetArrayFromImage(image)
    if label_array.ndim != 3:
        raise ValueError(f"Expected a 3D label volume, got shape {label_array.shape}: {path}")
    if not np.issubdtype(label_array.dtype, np.integer):
        rounded = np.rint(label_array)
        if not np.allclose(label_array, rounded):
            raise ValueError(f"Label image contains non-integer values: {path}")
        label_array = rounded.astype(np.int16)
    return image, label_array


def spacing_zyx(image: sitk.Image) -> tuple[float, float, float]:
    spacing_xyz = image.GetSpacing()
    if len(spacing_xyz) != 3:
        raise ValueError(f"Expected 3D image spacing, got: {spacing_xyz}")
    return float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0])


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    volume_a = int(mask_a.sum())
    volume_b = int(mask_b.sum())
    if volume_a == 0 and volume_b == 0:
        return 1.0
    if volume_a == 0 or volume_b == 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    return 2.0 * intersection / (volume_a + volume_b)


def surface_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, ~eroded)


def crop_to_union(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    margin_vox: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    union_mask = np.logical_or(mask_a, mask_b)
    if not union_mask.any():
        return mask_a, mask_b
    coordinates = np.argwhere(union_mask)
    lower = np.maximum(coordinates.min(axis=0) - margin_vox, 0)
    upper = np.minimum(coordinates.max(axis=0) + margin_vox + 1, mask_a.shape)
    crop_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    return mask_a[crop_slices], mask_b[crop_slices]


def surface_distances_mm(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    voxel_spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    if not mask_a.any() and not mask_b.any():
        return np.array([0.0], dtype=np.float32)
    if not mask_a.any() or not mask_b.any():
        return np.array([np.inf], dtype=np.float32)

    mask_a, mask_b = crop_to_union(mask_a, mask_b)
    surface_a = surface_mask(mask_a)
    surface_b = surface_mask(mask_b)
    distance_to_b = ndi.distance_transform_edt(~surface_b, sampling=voxel_spacing_zyx)
    distance_to_a = ndi.distance_transform_edt(~surface_a, sampling=voxel_spacing_zyx)
    return np.concatenate([distance_to_b[surface_a], distance_to_a[surface_b]])


def hd95_mm(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    voxel_spacing_zyx: tuple[float, float, float],
) -> float:
    distances = surface_distances_mm(mask_a, mask_b, voxel_spacing_zyx)
    return float(np.percentile(distances, 95))


def assd_mm(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    voxel_spacing_zyx: tuple[float, float, float],
) -> float:
    distances = surface_distances_mm(mask_a, mask_b, voxel_spacing_zyx)
    return float(np.mean(distances))


def fleiss_kappa(label_arrays: list[np.ndarray]) -> float:
    stacked = np.stack(label_arrays, axis=0)
    n_raters = stacked.shape[0]
    flattened = stacked.reshape(n_raters, -1)
    foreground_voxels = np.any(flattened > 0, axis=0)
    flattened = flattened[:, foreground_voxels]
    if flattened.shape[1] == 0:
        return float("nan")

    labels = np.unique(flattened)
    counts = np.zeros((flattened.shape[1], labels.size), dtype=np.float64)
    for label_index, label_value in enumerate(labels):
        counts[:, label_index] = np.sum(flattened == label_value, axis=0)

    p_j = counts.sum(axis=0) / (counts.shape[0] * n_raters)
    p_i = (np.sum(counts * counts, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))
    p_e = float(np.sum(p_j * p_j))
    if np.isclose(1.0 - p_e, 0.0):
        return 1.0 if np.isclose(p_bar, 1.0) else float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def finite_mean(values: list[object]) -> float:
    numeric_values = np.asarray(values, dtype=np.float64)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def finite_pstdev(values: list[object]) -> float:
    numeric_values = np.asarray(values, dtype=np.float64)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size <= 1:
        return 0.0
    return float(pstdev(finite_values.tolist()))


def find_label_files(label_dir: Path) -> list[Path]:
    return sorted(path for path in label_dir.rglob("*") if path.is_file() and is_nifti_path(path))


def output_stem(path: Path) -> str:
    return strip_nifti_suffix(path.name)


def doctor_title(rater_name: str) -> str:
    return "Doctor " + rater_name.split("_")[-1]


def literature_cell(row: dict[str, object], mean_key: str, std_key: str, markdown: bool) -> str:
    plus_minus = "±" if markdown else r"$\pm$"
    return f"{float(row[mean_key]):.3f} {plus_minus} {float(row[std_key]):.3f}"


def mean_literature_cell(rows: list[dict[str, object]], mean_key: str, markdown: bool) -> str:
    plus_minus = "±" if markdown else r"$\pm$"
    values = [row[mean_key] for row in rows]
    return f"{finite_mean(values):.3f} {plus_minus} {finite_pstdev(values):.3f}"


def write_wide_table(path: Path, rows: list[dict[str, object]]) -> None:
    labels = sorted({int(row["label"]) for row in rows})
    raters = sorted({str(row["rater"]) for row in rows})
    lookup = {(int(row["label"]), str(row["rater"])): row for row in rows}

    fieldnames = ["label"]
    for rater_name in raters:
        fieldnames.extend([f"{rater_name}_dice", f"{rater_name}_hd95_mm"])

    wide_rows = []
    for label_value in labels:
        wide_row: dict[str, object] = {"label": label_value}
        for rater_name in raters:
            row = lookup[(label_value, rater_name)]
            wide_row[f"{rater_name}_dice"] = row["mean_dice"]
            wide_row[f"{rater_name}_hd95_mm"] = row["mean_hd95_mm"]
        wide_rows.append(wide_row)
    write_csv(path, wide_rows)


def write_literature_tables(output_dir: Path, rows: list[dict[str, object]]) -> None:
    labels = sorted({int(row["label"]) for row in rows})
    raters = sorted({str(row["rater"]) for row in rows})
    lookup = {(int(row["label"]), str(row["rater"])): row for row in rows}
    label_groups = {
        label: [lookup[(label, rater)] for rater in raters]
        for label in labels
    }
    rater_groups = {
        rater: [lookup[(label, rater)] for label in labels]
        for rater in raters
    }

    def build_markdown_table(
        title: str,
        description: str,
        mean_key: str,
        std_key: str,
    ) -> list[str]:
        table_lines = [
            f"## {title}",
            "",
            description,
            "",
            (
                "| Substructure Label | "
                + " | ".join(doctor_title(rater) for rater in raters)
                + " | Mean |"
            ),
            "|" + "---|" * (len(raters) + 2),
        ]
        for label_value in labels:
            cells = [f"Label {label_value}"]
            cells.extend(
                literature_cell(lookup[(label_value, rater)], mean_key, std_key, markdown=True)
                for rater in raters
            )
            cells.append(mean_literature_cell(label_groups[label_value], mean_key, markdown=True))
            table_lines.append("| " + " | ".join(cells) + " |")
        mean_cells = ["Mean"]
        mean_cells.extend(
            mean_literature_cell(rater_groups[rater], mean_key, markdown=True)
            for rater in raters
        )
        mean_cells.append(mean_literature_cell(rows, mean_key, markdown=True))
        table_lines.append("| " + " | ".join(mean_cells) + " |")
        return table_lines

    dice_table = build_markdown_table(
        title="Table 1. Dice Consistency Across Six Doctors",
        description=(
            "Values are reported as Dice ± SD. Each value summarizes the agreement "
            "between each doctor's annotation and the reference annotation across "
            "available cases for the corresponding vertebral substructure label."
        ),
        mean_key="mean_dice",
        std_key="std_dice",
    )
    hd95_table = build_markdown_table(
        title="Table 2. HD95 Consistency Across Six Doctors",
        description=(
            "Values are reported as HD95 ± SD in millimeters. Each value summarizes "
            "the surface-distance agreement between each doctor's annotation and the "
            "reference annotation across available cases for the corresponding "
            "vertebral substructure label."
        ),
        mean_key="mean_hd95_mm",
        std_key="std_hd95_mm",
    )
    note_lines = [
        "",
        (
            "Note: Dice is dimensionless; HD95 is in mm. The reference annotation "
            "is used as the common comparison target for the six doctors."
        ),
    ]
    markdown_lines = [
        "# Inter-observer Annotation Consistency Across Six Doctors",
        "",
        *dice_table,
        "",
        *hd95_table,
        *note_lines,
    ]
    (output_dir / "baseline_comparison_literature_table.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "baseline_comparison_dice_table.md").write_text(
        "\n".join(dice_table + note_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "baseline_comparison_hd95_table.md").write_text(
        "\n".join(hd95_table + note_lines) + "\n",
        encoding="utf-8",
    )

    latex_break = r" \\"

    def build_latex_table(
        caption: str,
        label: str,
        mean_key: str,
        std_key: str,
    ) -> list[str]:
        table_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l" + "c" * (len(raters) + 1) + r"}",
            r"\hline",
            (
                "Substructure"
                + " & "
                + " & ".join(doctor_title(rater) for rater in raters)
                + " & Mean"
                + latex_break
            ),
            r"\hline",
        ]
        for label_value in labels:
            cells = [f"Label {label_value}"]
            cells.extend(
                literature_cell(lookup[(label_value, rater)], mean_key, std_key, markdown=False)
                for rater in raters
            )
            cells.append(mean_literature_cell(label_groups[label_value], mean_key, markdown=False))
            table_lines.append(" & ".join(cells) + latex_break)
        mean_cells = ["Mean"]
        mean_cells.extend(
            mean_literature_cell(rater_groups[rater], mean_key, markdown=False)
            for rater in raters
        )
        mean_cells.append(mean_literature_cell(rows, mean_key, markdown=False))
        table_lines.append(" & ".join(mean_cells) + latex_break)
        table_lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table*}"])
        return table_lines

    dice_latex = build_latex_table(
        caption=(
            r"Dice agreement between the reference annotation and six doctors. "
            r"Values are reported as Dice $\pm$ SD."
        ),
        label="tab:dice_doctor_consistency",
        mean_key="mean_dice",
        std_key="std_dice",
    )
    hd95_latex = build_latex_table(
        caption=(
            r"HD95 agreement between the reference annotation and six doctors. "
            r"Values are reported as HD95 $\pm$ SD in millimeters."
        ),
        label="tab:hd95_doctor_consistency",
        mean_key="mean_hd95_mm",
        std_key="std_hd95_mm",
    )
    latex_lines = [*dice_latex, "", *hd95_latex]
    (output_dir / "baseline_comparison_literature_table.tex").write_text(
        "\n".join(latex_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "baseline_comparison_dice_table.tex").write_text(
        "\n".join(dice_latex) + "\n",
        encoding="utf-8",
    )
    (output_dir / "baseline_comparison_hd95_table.tex").write_text(
        "\n".join(hd95_latex) + "\n",
        encoding="utf-8",
    )


def validate_case(
    case_id: str,
    rater_to_array: dict[str, np.ndarray],
    check_foreground_union: bool,
) -> np.ndarray:
    arrays = list(rater_to_array.values())
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Shape mismatch in case {case_id}: {shapes}")

    unique_sets = {tuple(np.unique(array).tolist()) for array in arrays}
    if len(unique_sets) != 1:
        raise ValueError(f"Label set mismatch in case {case_id}: {unique_sets}")

    if check_foreground_union:
        first_foreground = arrays[0] > 0
        for rater_name, array in rater_to_array.items():
            if not np.array_equal(first_foreground, array > 0):
                raise ValueError(f"Foreground union mismatch in case {case_id}, rater {rater_name}")

    return np.array(unique_sets.pop(), dtype=np.int16)


def summarize_pairwise_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), int(row["label"]))].append(row)

    summary_rows = []
    for (case_id, label_value), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "case_id": case_id,
                "label": label_value,
                "mean_dice": finite_mean([row["dice"] for row in group]),
                "mean_hd95_mm": finite_mean([row["hd95_mm"] for row in group]),
                "mean_assd_mm": finite_mean([row["assd_mm"] for row in group]),
                "num_pairs": len(group),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    label_files = find_label_files(args.baseline_label_dir)
    rater_dirs = find_rater_dirs(args.annotations_dir, args.raters)
    rater_names = [path.name for path in rater_dirs]
    if not label_files:
        raise FileNotFoundError(f"No baseline NIfTI label files found in: {args.baseline_label_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_metric_rows: list[dict[str, object]] = []
    summary_groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)

    for case_index, baseline_path in enumerate(label_files):
        reference_image, baseline_label = read_label(baseline_path)
        relative_parent = baseline_path.relative_to(args.baseline_label_dir).parent
        case_id = str(relative_parent / output_stem(baseline_path)).replace("\\", "/")
        case_stem = output_stem(baseline_path)

        labels = np.unique(baseline_label)
        if not args.include_background:
            labels = labels[labels > 0]
        voxel_spacing_zyx = spacing_zyx(reference_image)

        for rater_dir in rater_dirs:
            rater_name = rater_dir.name
            rater_path = rater_dir / relative_parent / f"{case_stem}_{rater_name}.nii.gz"
            if not rater_path.exists():
                matches = sorted((rater_dir / relative_parent).glob(f"{case_stem}_{rater_name}.nii*"))
                if not matches:
                    raise FileNotFoundError(f"Missing rater label: {rater_path}")
                rater_path = matches[0]

            _, rater_label = read_label(rater_path)
            if baseline_label.shape != rater_label.shape:
                raise ValueError(
                    f"Shape mismatch for {case_id}, {rater_name}: "
                    f"{baseline_label.shape} vs {rater_label.shape}"
                )
            if not np.array_equal(np.unique(baseline_label), np.unique(rater_label)):
                raise ValueError(f"Label set mismatch for {case_id}, {rater_name}")
            if args.check_foreground_union and not np.array_equal(
                baseline_label > 0,
                rater_label > 0,
            ):
                raise ValueError(f"Foreground union mismatch for {case_id}, {rater_name}")

            for label_value in labels:
                baseline_mask = baseline_label == label_value
                rater_mask = rater_label == label_value
                row = {
                    "case_id": case_id,
                    "rater": rater_name,
                    "label": int(label_value),
                    "dice": dice_score(baseline_mask, rater_mask),
                    "hd95_mm": hd95_mm(baseline_mask, rater_mask, voxel_spacing_zyx),
                }
                case_metric_rows.append(row)
                summary_groups[(int(label_value), rater_name)].append(row)

        print(f"[{case_index + 1}/{len(label_files)}] {case_id}")

    summary_rows = []
    for (label_value, rater_name), rows in sorted(summary_groups.items()):
        dice_values = [row["dice"] for row in rows]
        hd95_values = [row["hd95_mm"] for row in rows]
        summary_rows.append(
            {
                "label": label_value,
                "rater": rater_name,
                "mean_dice": finite_mean(dice_values),
                "std_dice": finite_pstdev(dice_values),
                "mean_hd95_mm": finite_mean(hd95_values),
                "std_hd95_mm": finite_pstdev(hd95_values),
                "num_cases": len(rows),
            }
        )

    write_csv(args.output_dir / "baseline_comparison_case_metrics.csv", case_metric_rows)
    write_csv(args.output_dir / "baseline_comparison_by_part_and_rater.csv", summary_rows)
    write_wide_table(args.output_dir / "baseline_comparison_table_wide.csv", summary_rows)
    write_literature_tables(args.output_dir, summary_rows)

    report = {
        "baseline_label_dir": str(args.baseline_label_dir),
        "annotations_dir": str(args.annotations_dir),
        "raters": rater_names,
        "num_cases": len(label_files),
        "num_case_metric_rows": len(case_metric_rows),
        "num_summary_rows": len(summary_rows),
        "mean_dice": finite_mean([row["dice"] for row in case_metric_rows]),
        "mean_hd95_mm": finite_mean([row["hd95_mm"] for row in case_metric_rows]),
    }
    with (args.output_dir / "baseline_comparison_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
