# SpineSubNet

Official PyTorch / [MONAI](https://monai.io/) implementation for **3D lumbar vertebra substructure segmentation** from grayscale CT volumes.

End-to-end pipeline: data preparation → 3D U-Net training → batch inference → evaluation → post-processing → annotation consistency analysis.

## Overview

| Item | Description |
|------|-------------|
| **Task** | Multi-class 3D segmentation of lumbar vertebral substructures |
| **Input** | Single-channel grayscale CT (`.nii` / `.nii.gz`) |
| **Output** | 8-class label map (background + 7 subregions) |
| **Model** | 3D U-Net ([MONAI](https://docs.monai.io/en/stable/networks.html#unet)) |
| **Loss** | Cross-Entropy + Dice |
| **Metrics** | Dice (DSC), Hausdorff Distance 95% (HD95) |

Each sample is one cropped lumbar level (e.g. `patient001_L3.nii.gz`). Subject-level splitting keeps all levels from the same patient in the same train / validation / test split.

## Repository Layout

```
SpineSubNet/
├── configs/          # YAML configs for training and inference
├── consistency/      # Sample data and inter-rater consistency evaluation
├── data/             # Transforms and DataLoader utilities
├── experiments/      # Pretrained model checkpoints
├── losses/           # Segmentation loss functions
├── models/           # U-Net factory and network definitions
├── scripts/          # Shell helpers (train / inference)
├── tools/            # CLI utilities (see Tools Reference)
├── trainers/         # Training loops
├── utils/            # Config parsing, logging, experiment setup
├── train.py          # Main training entry point
└── requirements.txt
```

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU recommended.

```bash
git clone https://github.com/<your-username>/SpineSubNet.git
cd SpineSubNet

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Checkpoints and sample NIfTI files use Git LFS. Run `git lfs pull` after cloning if files appear as pointer stubs.

## Data Preparation

### Dataset layout

```
data/subregion/
├── binary/       # Binary foreground masks (same filenames)
├── grayscale/    # Grayscale CT volumes
├── label/        # Multi-class subregion labels (classes 0–7)
└── dataset.json  # Generated below
```

- Filenames include a lumbar level suffix, e.g. `patient001_L3.nii.gz`
- `binary/`, `grayscale/`, and `label/` must share matching filenames
- Training uses **grayscale** and **label** only; `binary/` is required for JSON generation

### Generate splits

`tools/create_dataset.py` scans the three folders, validates file pairs, and writes a Decathlon-style `dataset.json`. Splits are grouped by subject prefix (text before `_L{level}`); default ratio is 70% / 10% / 20%.

```bash
python tools/create_dataset.py \
  --data_dir /path/to/your_dataset \
  --output dataset.json
```

Optional: `--train_ratio`, `--val_ratio`, `--test_ratio`, `--seed`.

### Crop lumbar regions

`tools/crop_lumbar_ct.py` extracts per-vertebra crops from full-spine CT using a vertebra segmentation mask. Label schemes: `verse` (L1–L6, IDs 20–25) or `colon` (L1–L5, IDs 20–24).

```bash
python tools/crop_lumbar_ct.py \
  --ct_dir /path/to/rawdata \
  --mask_dir /path/to/derivatives \
  --output_dir /path/to/output \
  --dataset verse \
  --margin 0
```

### Check CT sizes

`tools/stat_ct_size.py` reports voxel shapes and spacings before and after the training resampling step (`Spacingd` to 1.0 mm). Useful for verifying the default 160³ patch size.

```bash
python tools/stat_ct_size.py --input_dir /path/to/grayscale
python tools/stat_ct_size.py --json dataset.json --data_root /path/to/data --split training
```

## Training

Config: `configs/baseline_grayscale.yaml`

| Setting | Default |
|---------|---------|
| Spatial size | 160 × 160 × 160 |
| Target spacing | 1.0 × 1.0 × 1.0 mm |
| Intensity window | HU [-500, 1500] → [0, 1] |
| Optimizer | AdamW (lr=1e-4), WarmupCosine schedule |
| Max iterations | 50,000 (validate every 500 steps) |

Preprocessing uses MONAI transforms: RAS orientation, isotropic resampling, HU windowing, spatial padding, and random flips / 90° rotations.

```bash
python train.py --config configs/baseline_grayscale.yaml

# Optional overrides
python train.py --config configs/baseline_grayscale.yaml \
  --data_dir /path/to/data --batch_size 1 --lr 0.0001

# Or
bash scripts/train_baseline.sh
```

Outputs under `experiments/baseline_grayscale/`:
- `weights/best_metric.pth` — best validation Dice
- `weights/checkpoint_step_*.pth` — periodic snapshots
- TensorBoard logs (if enabled)

## Inference

Pretrained checkpoints in `experiments/`:

| Checkpoint | Trained on |
|------------|------------|
| `Colon_best_metric.pth` | Colon |
| `Verse_best_metric.pth` | VerSe |
| `LumASe_best_metric.pth` | LumASe |
| `Total_best_metric.pth` | Combined (Total) |

Edit `configs/inference.yaml` — set `model.checkpoint`, `inference.input_dir`, and `inference.output_dir`. Batch inference applies the same preprocessing as validation and writes `*_seg.nii.gz` files. Set `inference.resample_to_original: true` to map predictions back to native geometry.

```bash
python tools/inference.py --config configs/inference.yaml
# or: bash scripts/inference.sh
```

## Evaluation

`tools/evaluate_model.py` runs inference on the **testing** split in `dataset.json` and reports overall and per-class Dice and HD95.

```bash
python tools/evaluate_model.py \
  --model_path experiments/baseline_grayscale/weights/best_metric.pth \
  --config configs/baseline_grayscale.yaml \
  --test_json /path/to/dataset.json \
  --output_dir evaluation_results/baseline
```

Use `--skip_hd95` to reduce GPU memory. Outputs: run log, per-sample CSV, and summary text with per-class statistics.

## Post-processing

`tools/process.py` aligns predictions to reference mask geometry in parallel. For each segmentation / mask pair:

1. Remove spurious connected components per label
2. Zero out voxels outside the reference foreground mask
3. Fill empty mask voxels with the nearest valid label

```bash
python tools/process.py \
  --seg_dir /path/to/segmentations \
  --mask_dir /path/to/reference_masks \
  --output_dir /path/to/aligned_output \
  --num_workers 8
```

Files are matched by filename (strips `_seg`, `_pred`, `_inference` suffixes when needed).

## Annotation Consistency

`consistency/evaluate_annotation_consistency.py` compares rater annotations against baseline reference labels (Dice and HD95 per label). Each rater is evaluated independently against the baseline.

```
consistency/
├── label/          # Baseline reference labels
├── anno/
│   ├── doctor_01/  # Rater files: {case}_{rater}.nii.gz
│   └── ...
└── grayscale/      # Optional CT volumes
```

```bash
python consistency/evaluate_annotation_consistency.py \
  --baseline-label-dir consistency/label \
  --annotations-dir consistency/anno \
  --output-dir consistency/results
```

Outputs: per-case CSV, per-label/per-rater summary, wide-format table, JSON summary, and LaTeX tables.

## Tools Reference

| Script | Purpose | Key arguments |
|--------|---------|---------------|
| `create_dataset.py` | Build `dataset.json` with subject-level splits | `--data_dir`, `--output`, `--seed` |
| `crop_lumbar_ct.py` | Crop per-vertebra CT from full-spine scans | `--ct_dir`, `--mask_dir`, `--dataset` |
| `stat_ct_size.py` | Summarize shapes/spacings before/after resampling | `--input_dir` or `--json` + `--data_root` |
| `inference.py` | Batch 3D U-Net inference | `--config` |
| `evaluate_model.py` | Test-set Dice / HD95 evaluation | `--model_path`, `--config`, `--test_json` |
| `process.py` | Post-process and align segmentations | `--seg_dir`, `--mask_dir`, `--output_dir` |

## Citation

If this repository is useful for your work, please cite the associated paper:

```bibtex
@article{topospinenet2026,
  title   = {A Knowledge Distillation Framework and Large-Scale Dataset for CT Segmentation of Lumbar Substructure},
  year    = {2026}
}
```

## License

See repository license file for terms of use.
