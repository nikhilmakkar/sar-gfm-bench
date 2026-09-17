"""DINOv2 backbone wrapper (Group A: standard ViT, RGB input).

Input adaptation: SAR [0,1] x3 channels -> ImageNet normalization.
Patch size 14. The wrapper resizes the input to the nearest multiple of 14
(512 -> 518) before the encoder; the adapter then maps features back to
exact strides relative to the ORIGINAL size, so detector geometry is
unaffected by the resize.

Feature extraction: encoder.get_intermediate_layers(x, n=<block indices>,
reshape=True) returns (B, D, h, w) spatial maps with CLS/register tokens
already stripped — exactly what ViTDetAdapter wants.
"""

import math

import torch
import torch.nn.functional as F
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

DINOV2_CONFIGS = {
    'vits14': dict(embed_dim=384, num_layers=12),
    'vitb14': dict(embed_dim=768, num_layers=12),
    'vitl14': dict(embed_dim=1024, num_layers=24),
    'vitg14': dict(embed_dim=1536, num_layers=40),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@MODELS.register_module()
class DINOv2Backbone(BaseGFMBackbone):

    PATCH = 14

    def __init__(self, variant='vitl14', frozen=True, out_channels=256,
                 tap_indices=None, bf16=False, tap_norm=True):
        # tap_norm: pass taps through the model's final LayerNorm (Meta's API
        # default; their classification probes) vs raw block outputs (what
        # Meta's own dense/depth heads use, norm=False). Ablation knob.
        super().__init__(frozen=frozen, bf16=bf16)
        self.tap_norm = tap_norm
        cfg = DINOV2_CONFIGS[variant]
        self.encoder = torch.hub.load(
            'facebookresearch/dinov2', f'dinov2_{variant}', pretrained=True)

        L = cfg['num_layers']
        # 0-based block indices at L/4, L/2, 3L/4, L
        self.tap_indices = tap_indices or [
            L // 4 - 1, L // 2 - 1, 3 * L // 4 - 1, L - 1]

        self.adapter = ViTDetAdapter(cfg['embed_dim'], out_channels)

        self.register_buffer(
            'pixel_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            'pixel_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def forward(self, x):
        H, W = x.shape[-2:]
        x = (x - self.pixel_mean) / self.pixel_std

        # nearest multiple of patch size (512 -> 518)
        Hp = max(self.PATCH, round(H / self.PATCH) * self.PATCH)
        Wp = max(self.PATCH, round(W / self.PATCH) * self.PATCH)
        if (Hp, Wp) != (H, W):
            x = F.interpolate(x, size=(Hp, Wp), mode='bilinear',
                              align_corners=False)

        with self.encoder_context():
            feats = self.encoder.get_intermediate_layers(
                x, n=self.tap_indices, reshape=True, norm=self.tap_norm)
        feats = [f.float() for f in feats]

        return self.adapter(feats, (H, W))
