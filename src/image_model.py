"""
Standalone face emotion classifier — Image only, no fusion, no text, no audio.

Proven TinyCNN architecture:
  3 × Conv2d(stride=2) + BN + ReLU → GAP → Linear(128 → 7)
  NO Dropout, stride-2 convs, direct classification.

Trained on FER2013 with balanced oversampling + SGD+Nesterov.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEmotionClassifier(nn.Module):
    """Stride-2 CNN — 94K params, val_acc=54.4%, test_acc=40.4% on FER2013."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # Layer 1: 48→24
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 2: 24→12
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 3: 12→6
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # → [B, 128, 1, 1]
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128, NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)           # [B, 128, 6, 6]
        feat = self.gap(feat)            # [B, 128, 1, 1]
        feat = self.flatten(feat)        # [B, 128]
        return self.classifier(feat)     # [B, 7]

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
