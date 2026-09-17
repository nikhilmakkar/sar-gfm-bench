"""ViTDet-style multi-scale adapter.

Takes 4 spatial feature maps tapped from a ViT encoder (all at the ViT's
native stride = patch_size) and produces 4 FPN-ready maps at exact strides
8/16/32/64 relative to the *original* input image.

Design notes:
- Wrappers hand us spatial maps (B, D, h, w), not token sequences. Each
  wrapper is responsible for stripping CLS/register tokens and reshaping,
  since those quirks are model-specific.
- We interpolate to exact target sizes (H//8, H//16, ...) computed from the
  input image shape rather than using scale_factor. With patch-14 models
  (512/14 = 36.6 grid) scale_factor would give fractional strides and
  RoIAlign coordinates would drift; exact sizes keep FPN strides honest.
"""

import torch.nn as nn
import torch.nn.functional as F

FPN_STRIDES = (8, 16, 32, 64)


class ViTDetAdapter(nn.Module):

    def __init__(self, embed_dim, out_channels=256):
        super().__init__()
        self.embed_dim = embed_dim
        self._out_channels = out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, out_channels, kernel_size=1),
                nn.GroupNorm(32, out_channels),
            ) for _ in FPN_STRIDES
        ])

    @property
    def out_channels(self):
        return [self._out_channels] * len(FPN_STRIDES)

    def forward(self, feats, input_hw):
        """
        Args:
            feats: list of 4 tensors (B, embed_dim, h, w), coarsest tap last.
            input_hw: (H, W) of the image batch fed to the encoder wrapper,
                used to compute exact FPN target sizes.
        Returns:
            tuple of 4 maps at strides 8/16/32/64.
        """
        assert len(feats) == len(FPN_STRIDES), \
            f'expected {len(FPN_STRIDES)} taps, got {len(feats)}'
        H, W = input_hw
        outs = []
        for feat, conv, stride in zip(feats, self.lateral_convs, FPN_STRIDES):
            target = (max(H // stride, 1), max(W // stride, 1))
            feat = conv(feat)
            if feat.shape[-2:] != target:
                feat = F.interpolate(
                    feat, size=target, mode='bilinear', align_corners=False)
            outs.append(feat)
        return tuple(outs)
