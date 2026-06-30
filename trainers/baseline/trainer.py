"""
Baseline Trainer for grayscale input training
"""

import os
import logging
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


class BaselineTrainer:
    """Trainer for baseline model (grayscale single-channel input)."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        root_dir: str = "./experiments",
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.root_dir = root_dir
        self.device = device

        from losses.segmentation import CombinedCEDiceLoss
        self.criterion = CombinedCEDiceLoss()

        out_channels = config.get('model', {}).get('out_channels', 8)
        self.post_label = AsDiscrete(to_onehot=out_channels)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=out_channels)
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean_batch",
            get_not_nans=False
        )

        self.global_step = 0
        self.current_epoch = 0
        self.best_dice = 0.0
        self.best_step = 0
        self.train_loss_history = []
        self.val_dice_history = []

        logging.info("BaselineTrainer initialized with grayscale input")
        logging.info(f"Device: {self.device}")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        max_iterations = self.config.get('max_iterations', 5000)
        eval_num = self.config.get('eval_num', 50)

        epoch_loss = 0.0
        step_count = 0

        epoch_iterator = tqdm(
            self.train_loader,
            desc=f"Training (Step {self.global_step}/{max_iterations})",
            dynamic_ncols=True
        )

        for batch in epoch_iterator:
            inputs = batch["grayscale"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(inputs)

            labels_squeezed = labels.squeeze(1).long()
            loss_dict = self.criterion(outputs, labels_squeezed)
            loss = loss_dict['total']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss += loss.item()
            step_count += 1

            epoch_iterator.set_description(
                f"Training (Step {self.global_step}/{max_iterations}) "
                f"(loss={loss.item():.5f})"
            )

            if self.writer is not None:
                self.writer.add_scalar("train/loss_total", loss.item(), self.global_step)
                self.writer.add_scalar("train/loss_ce", loss_dict['ce'].item(), self.global_step)
                self.writer.add_scalar("train/loss_dice", loss_dict['dice'].item(), self.global_step)

                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)

            if (self.global_step % eval_num == 0 and self.global_step != 0) or \
               self.global_step == max_iterations:
                avg_train_loss = epoch_loss / step_count
                self.validate(avg_train_loss)
                self.model.train()

            if self.global_step % 1000 == 0 and self.global_step != 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")

            self.global_step += 1

            if self.global_step >= max_iterations:
                break

        return epoch_loss / step_count if step_count > 0 else 0.0

    def validate(self, train_loss: float = 0.0) -> Tuple[float, np.ndarray]:
        """Run validation."""
        self.model.eval()

        val_iterator = tqdm(
            self.val_loader,
            desc=f"Validation (Step {self.global_step})",
            dynamic_ncols=True
        )

        with torch.no_grad():
            for batch in val_iterator:
                inputs = batch["grayscale"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)

                try:
                    from monai.data import MetaTensor
                    if isinstance(labels, MetaTensor):
                        labels = labels.as_tensor()
                    if isinstance(outputs, MetaTensor):
                        outputs = outputs.as_tensor()
                except ImportError:
                    pass

                labels_list = decollate_batch(labels)
                outputs_list = decollate_batch(outputs)

                labels_converted = [self.post_label(label) for label in labels_list]
                outputs_converted = [self.post_pred(output) for output in outputs_list]

                self.dice_metric(y_pred=outputs_converted, y=labels_converted)

        dice_per_class = self.dice_metric.aggregate()
        if isinstance(dice_per_class, torch.Tensor):
            dice_per_class = dice_per_class.cpu().numpy()

        mean_dice = float(np.mean(dice_per_class))
        self.dice_metric.reset()

        logging.info(f"\n{'='*70}")
        logging.info(f"Validation Results (Step: {self.global_step})")
        logging.info(f"{'='*70}")
        logging.info(f"Average Training Loss: {train_loss:.5f}")
        logging.info(f"Mean Dice Score: {mean_dice:.4f}")
        logging.info(f"\nDice Scores per Class:")
        for idx, dice_score in enumerate(dice_per_class):
            logging.info(f"  Class {idx}: {dice_score:.4f}")
        logging.info(f"{'='*70}")

        if self.writer is not None:
            self.writer.add_scalar("val/mean_dice", mean_dice, self.global_step)
            for idx, dice_score in enumerate(dice_per_class):
                self.writer.add_scalar(f"val/dice_class_{idx}", dice_score, self.global_step)

        if mean_dice > self.best_dice:
            self.best_dice = mean_dice
            self.best_step = self.global_step
            self.save_checkpoint("best_metric.pth")
            logging.info(
                f"\n✓ Best model saved! "
                f"Best Mean Dice: {self.best_dice:.4f} | "
                f"Current Mean Dice: {mean_dice:.4f}\n"
            )
        else:
            logging.info(
                f"\n✗ Model not saved (not best). "
                f"Best Mean Dice: {self.best_dice:.4f} | "
                f"Current Mean Dice: {mean_dice:.4f}\n"
            )

        self.val_dice_history.append(mean_dice)
        return mean_dice, dice_per_class

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        weights_dir = os.path.join(self.root_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        checkpoint_path = os.path.join(weights_dir, filename)
        torch.save(self.model.state_dict(), checkpoint_path)

        if "best" in filename:
            logging.info(f"✓ Best model saved: {checkpoint_path}")
        else:
            logging.info(f"✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        )
        logging.info(f"✓ Checkpoint loaded: {checkpoint_path}")

    def train(self) -> None:
        """Main training loop."""
        max_iterations = self.config.get('max_iterations', 5000)

        logging.info("\n" + "=" * 70)
        logging.info("Starting Baseline Training (Grayscale)")
        logging.info("=" * 70)
        logging.info(f"Max iterations: {max_iterations}")
        logging.info(f"Validation interval: every {self.config.get('eval_num', 50)} steps\n")

        torch.backends.cudnn.benchmark = True

        while self.global_step < max_iterations:
            epoch_loss = self.train_epoch()
            self.train_loss_history.append(epoch_loss)
            self.current_epoch += 1

        self.save_checkpoint("final_model.pth")

        logging.info("\n" + "=" * 70)
        logging.info("Training Completed!")
        logging.info("=" * 70)
        logging.info(f"✓ Best Dice Score: {self.best_dice:.4f}")
        logging.info(f"✓ Best step: {self.best_step}")
        logging.info(f"✓ Final model saved to: {os.path.join(self.root_dir, 'weights', 'final_model.pth')}")
        logging.info(f"✓ Best model saved to: {os.path.join(self.root_dir, 'weights', 'best_metric.pth')}")
        logging.info(f"✓ Experiment directory: {self.root_dir}")

        if self.writer is not None:
            self.writer.close()
