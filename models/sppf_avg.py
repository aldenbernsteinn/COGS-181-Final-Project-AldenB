"""SPPF variant using AvgPool2d instead of MaxPool2d.

Tests Task.md item (d): different pooling functions.
Drop-in replacement for ultralytics SPPF module.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class SPPFAvg(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPFAvg) with average pooling."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
