"""Base class for all GFM backbone wrappers.

Input contract (identical for every backbone in the benchmark):
    forward(x) receives (B, 3, H, W) float in [0, 1] — the SAR amplitude
    replicated across 3 channels by the data pipeline. Each wrapper adapts
    this to whatever its encoder expects (ImageNet norm, single band +
    wavelength metadata, ...). Model-specific input handling lives HERE,
    never in the shared data pipeline — that keeps the pipeline identical
    across all 12 models.

Output contract:
    tuple of 4 feature maps at exact strides 8/16/32/64 (via ViTDetAdapter),
    each with `out_channels` channels, ready for mmdet's FPN neck.

Freezing:
    frozen=True freezes encoder params AND pins the encoder in eval() mode
    (train() is overridden), while the adapter stays trainable. The encoder
    forward runs under no_grad to save memory.
"""

from abc import abstractmethod
from contextlib import contextmanager

import torch
import torch.nn as nn


class BaseGFMBackbone(nn.Module):

    def __init__(self, frozen=True, bf16=False):
        super().__init__()
        self.frozen = frozen
        self.bf16 = bf16

    @abstractmethod
    def forward(self, x):
        ...

    def freeze_encoder(self, encoder):
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen and hasattr(self, 'encoder'):
            self.encoder.eval()
        return self

    @contextmanager
    def encoder_context(self):
        # bf16 autocast covers the encoder forward only; wrappers cast their
        # outputs back to fp32, so adapter/FPN/heads always train in fp32.
        grad_ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with grad_ctx, torch.autocast('cuda', dtype=torch.bfloat16,
                                      enabled=self.bf16):
            yield

    def log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[{type(self).__name__}] params total {total/1e6:.1f}M, '
              f'trainable {trainable/1e6:.1f}M')
