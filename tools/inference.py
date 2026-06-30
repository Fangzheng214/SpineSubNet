"""
Vertebra Subregion Segmentation Inference Script

Batch grayscale baseline inference with optional resampling back to native NIfTI geometry.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, Any
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from monai.transforms import Compose, SaveImaged

from models import create_model
from data.transforms import _build_base_transforms


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"[OK] Config loaded: {config_path}")
    return config


def get_inference_transforms(
    target_spacing: tuple,
    spatial_size: tuple,
    use_spacing: bool,
) -> Compose:
    """Build inference transforms for grayscale input."""
    keys = ["grayscale"]
    base_transforms = _build_base_transforms(
        keys=keys,
        spatial_size=spatial_size,
        target_spacing=target_spacing,
        use_spacing=use_spacing,
    )
    return Compose(base_transforms)


def main():
    parser = argparse.ArgumentParser(
        description="Grayscale Baseline Segmentation Inference (batch mode)",
        epilog="Example: python tools/inference.py --config configs/inference.yaml"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Inference config file (e.g., configs/inference.yaml)")

    args = parser.parse_args()

    print("=" * 70)
    print("Grayscale Baseline Segmentation Inference")
    print("=" * 70)

    config = load_config(args.config)

    model_config = config.get('model', {})
    data_config = config.get('data', {})
    inference_config = config.get('inference', {})

    checkpoint_path = model_config.get('checkpoint')
    spatial_size = tuple(data_config.get('spatial_size', [160, 160, 160]))
    target_spacing = tuple(data_config.get('target_spacing', [1.0, 1.0, 1.0]))
    use_spacing = data_config.get('use_spacing', True)

    input_dir = inference_config.get('input_dir')
    output_dir = inference_config.get('output_dir', './inference_output')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nConfiguration:")
    print(f"  Input dir:  {input_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Spatial size: {spatial_size}")
    print(f"  Target spacing: {target_spacing}")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model_arch_config = {
        'type': 'unet',
        'in_channels': 1,
        'out_channels': 8,
        'spatial_dims': 3,
    }

    val_transforms = get_inference_transforms(
        target_spacing, spatial_size, use_spacing
    )

    model = create_model(model_config=model_arch_config, img_size=spatial_size)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "unet" in checkpoint and isinstance(checkpoint["unet"], dict):
        state_dict = checkpoint["unet"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len("unet."):] if k.startswith("unet.") else k
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("[OK] Model loaded\n")

    print("=" * 70)
    print("Starting Batch Inference")
    print("=" * 70)
    print(f"{input_dir} → {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))

    if not image_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    print(f"Found {len(image_files)} files\n")

    output_saver = SaveImaged(
        output_dir=output_dir,
        output_postfix="_seg",
        output_ext=".nii.gz",
        resample=True,
        keys="pred",
    )

    success = 0
    with torch.no_grad():
        for idx, image_path in enumerate(image_files, 1):
            try:
                filename = os.path.basename(image_path)
                base_name = filename[:-7] if filename.endswith('.nii.gz') else os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}_seg.nii.gz")

                print(f"[{idx}/{len(image_files)}] {filename}")

                transformed_data = val_transforms({"grayscale": image_path})
                val_inputs = transformed_data["grayscale"].unsqueeze(0).to(device)

                val_outputs = model(val_inputs)
                val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True).cpu().to(torch.int16)

                img_tensor = transformed_data["grayscale"]
                if not hasattr(img_tensor, "meta"):
                    raise RuntimeError("No meta information found on grayscale tensor.")
                meta = img_tensor.meta.copy()
                meta["filename_or_obj"] = output_path

                output_saver({
                    "pred": val_outputs[0],
                    "image_meta_dict": meta,
                })

                print(f"  [OK] {output_path}\n")
                success += 1

            except Exception as e:
                print(f"  [FAIL] Failed: {repr(e)}")
                traceback.print_exc()
                print()

    print("=" * 70)
    print(f"Completed: {success}/{len(image_files)} successful")
    print("=" * 70)


if __name__ == "__main__":
    main()
