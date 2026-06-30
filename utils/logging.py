"""
Logging and experiment setup utilities
"""

import os
import sys
import logging
from datetime import datetime
from typing import Tuple

from torch.utils.tensorboard import SummaryWriter


def setup_logging(log_dir: str, experiment_name: str) -> None:
    """
    Setup logging to both console and file
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of experiment for log filename
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Log file: {log_filename}")


def setup_experiment_directory(
    root_dir: str,
    experiment_name: str,
    config_path: str = None
) -> Tuple[SummaryWriter, str]:
    """
    Create experiment directory with timestamp and initialize TensorBoard writer
    
    Args:
        root_dir: Root directory for experiments
        experiment_name: Name of experiment
        config_path: Path to config file (for copying)
    
    Returns:
        writer: TensorBoard writer
        full_root_dir: Full path to experiment directory
    """
    from .misc import copy_code_to_experiment
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_root_dir = os.path.join(root_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(full_root_dir, exist_ok=True)
    logging.info(f"Experiment directory: {full_root_dir}")
    
    # Setup logging
    log_dir = os.path.join(full_root_dir, "logs")
    setup_logging(log_dir, experiment_name)
    
    # Copy code for reproducibility
    copy_code_to_experiment(full_root_dir, config_path)
    
    # Initialize TensorBoard
    tensorboard_dir = os.path.join(full_root_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logging.info(f"TensorBoard directory: {tensorboard_dir}")
    
    return writer, full_root_dir

