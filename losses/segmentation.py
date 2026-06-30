"""
Segmentation loss functions (CE + Dice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation."""

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        if targets.ndim == logits.ndim - 1:
            targets = F.one_hot(targets, num_classes=logits.shape[1])
            targets = targets.permute(0, 4, 1, 2, 3).float()

        probs = probs.contiguous().view(probs.shape[0], probs.shape[1], -1)
        targets = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)

        intersection = (probs * targets).sum(dim=2)
        union = probs.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == 'mean':
            return 1.0 - dice.mean()
        if self.reduction == 'sum':
            return (1.0 - dice).sum()
        return 1.0 - dice


class CombinedCEDiceLoss(nn.Module):
    """Combined Cross-Entropy and Dice loss for segmentation."""

    def __init__(self, dice_smooth: float = 1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return {
            'total': loss_ce + loss_dice,
            'ce': loss_ce,
            'dice': loss_dice,
        }
