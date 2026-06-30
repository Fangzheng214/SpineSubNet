"""
Miscellaneous utility functions
"""

import os
import sys
import shutil
import logging

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logging.info(f"Random seed set to: {seed}")


def copy_code_to_experiment(root_dir: str, config_path: str = None) -> None:
    """Copy entire project code to experiment directory for reproducibility."""
    code_dir = os.path.join(root_dir, "code_snapshot")
    os.makedirs(code_dir, exist_ok=True)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    exclude_dirs = {
        'experiments',
        '__pycache__',
        '.git',
        '.vscode',
        '.idea',
        'venv',
        'env',
        '.pytest_cache',
        '.mypy_cache',
        'build',
        'dist',
        '*.egg-info',
    }

    exclude_extensions = {
        '.pyc',
        '.pyo',
        '.pyd',
        '.so',
        '.dll',
        '.dylib',
        '.log',
    }

    copied_count = 0
    skipped_count = 0

    logging.info(f"Copying project code from {project_root} to {code_dir}")

    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

        rel_path = os.path.relpath(root, project_root)

        if rel_path.startswith('experiments'):
            skipped_count += len(files)
            continue

        target_dir = code_dir if rel_path == '.' else os.path.join(code_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for filename in files:
            if any(filename.endswith(ext) for ext in exclude_extensions):
                skipped_count += 1
                continue

            if filename.startswith('.'):
                skipped_count += 1
                continue

            src_file = os.path.join(root, filename)
            dst_file = os.path.join(target_dir, filename)

            try:
                shutil.copy2(src_file, dst_file)
                copied_count += 1
            except Exception as e:
                logging.warning(f"Failed to copy {src_file}: {e}")
                skipped_count += 1

    logging.info(f"Code snapshot complete: {copied_count} files copied, {skipped_count} files skipped")

    if config_path and os.path.exists(config_path):
        config_filename = os.path.basename(config_path)
        dst = os.path.join(code_dir, config_filename)
        try:
            shutil.copy2(config_path, dst)
            logging.info(f"Copied config: {config_filename}")
        except Exception as e:
            logging.warning(f"Failed to copy config: {e}")

    cmd_file = os.path.join(code_dir, "command.txt")
    try:
        with open(cmd_file, 'w') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')
        logging.info("Saved command to: command.txt")
    except Exception as e:
        logging.warning(f"Failed to save command: {e}")
