import copy
import math
import timm
from functools import partial
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Sequence, Union, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import Conv2dNormActivation, StochasticDepth, SqueezeExcitation

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)

        self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return logits