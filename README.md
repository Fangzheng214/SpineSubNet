# SpineSubNet

Official PyTorch / [MONAI](https://monai.io/) implementation for **3D lumbar substructure segmentation** from grayscale CT volumes.

This repository provides a modular training and inference pipeline for vertebra subregion segmentation, including dataset preparation, baseline 3D U-Net training, batch inference with native-geometry resampling, quantitative evaluation, and post-processing utilities.

## Features

- **Training** — 1-channel 3D U-Net on grayscale CT input (`configs/baseline_grayscale.yaml`)
- **Data** — NIfTI (`.nii` / `.nii.gz`), Decathlon-style `dataset.json`, resampling and augmentations via MONAI
- **Inference** — Batch inference on a directory of NIfTI volumes with resampling back to native geometry
- **Evaluation** — Dice and HD95 on a held-out test set
- **Post-processing** — Parallel mask alignment against reference geometry (`tools/process.py`)

## Repository Layout

```
SpineSubNet/
├── configs/          # YAML experiment files
├── data/             # Data transforms and loaders
├── losses/           # Segmentation losses
├── models/           # Network definitions
├── scripts/          # Bash helpers
├── tools/            # Inference, evaluation, post-processing
├── trainers/         # Training loops
├── utils/            # Logging, config, experiment utilities
├── train.py          # Main training entry point
└── requirements.txt  # Python dependencies
```

## Installation

```bash
git clone https://github.com/<your-username>/SpineSubNet.git
cd SpineSubNet

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Format

Under one root directory (e.g. `data/subregion/`):

- `grayscale/` — intensity volumes (`.nii.gz`)
- `label/` — same filenames, multi-class subregion labels

Generate a `dataset.json` with train / validation / test splits:

```bash
python tools/create_dataset.py --data_dir /path/to/your_dataset --output dataset.json
```

## Training

```bash
python train.py --config configs/baseline_grayscale.yaml
```

Optional overrides:

```bash
python train.py --config configs/baseline_grayscale.yaml --data_dir /path/to/data --batch_size 1 --lr 0.0001
```

Or use the shell script:

```bash
bash scripts/train_baseline.sh
```

## Inference

Edit `configs/inference.yaml` (checkpoint path, input/output directories), then:

```bash
python tools/inference.py --config configs/inference.yaml
```

## Evaluation

```bash
python tools/evaluate_model.py \
  --model_path experiments/baseline_grayscale/weights/best_metric.pth \
  --config configs/baseline_grayscale.yaml \
  --test_json /path/to/your_dataset/dataset.json \
  --output_dir evaluation_results/baseline
```

## Post-processing

Align predicted segmentation masks to reference mask geometry:

```bash
python tools/process.py \
  --seg_dir /path/to/segmentations \
  --mask_dir /path/to/reference_masks \
  --output_dir /path/to/aligned_output
```

## Citation

If this repository is useful for your work, please cite the associated paper.

```bibtex
@article{topospinenet2026,
  title   = {A Knowledge Distillation Framework and Large-Scale Dataset for CT Segmentation of Lumbar Substructure},
  year    = {2026}
}
```
