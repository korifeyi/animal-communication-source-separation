import copy
import math
import timm
from functools import partial
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Sequence, Union, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import Conv2dNormActivation, StochasticDepth, SqueezeExcitation

class NormalizeMelSpec(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V


class EfficientVIT(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.normalizator = NormalizeMelSpec()
        self.backbone = timm.create_model("efficientvit_b0", pretrained=True, num_classes=num_classes, in_chans=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(self.normalizator(x))
        return logits