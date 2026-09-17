"""DINOv3 backbone wrapper (Group A: standard ViT, RGB input).

Two pretraining corpora, same architecture (ViT-L/16):
  - variant='vitl16_web' : LVD-1689M web images  (ImageNet norm)
  - variant='vitl16_sat' : SAT-493M Maxar ~0.6m overhead RGB (its own norm)

Patch size 16 divides 512 exactly — no input resize (unlike DINOv2/14).
Model code comes from the vendored dinov3 package in the DINO_Soars repository
(the official repo layout); weights are the license-gated HF checkpoints,
passed in via `checkpoint` as a local path.
"""

import os
import sys

import torch
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

DINOV3_ROOT = os.environ.get('DINOV3_ROOT', '../DINO_Soars')

# (builder kwargs, normalization) per variant. SAT-493M uses the satellite
# stats from Meta's model card, not ImageNet.
DINOV3_CONFIGS = {
    'vitl16_web': dict(weights='LVD1689M', embed_dim=1024, num_layers=24,
                       mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'vitl16_sat': dict(weights='SAT493M', embed_dim=1024, num_layers=24,
                       mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
}


@MODELS.register_module()
class DINOv3Backbone(BaseGFMBackbone):

    PATCH = 16

    def __init__(self, variant='vitl16_web', checkpoint=None, frozen=True,
                 out_channels=256, tap_indices=None, bf16=False, tap_norm=True):
        super().__init__(frozen=frozen, bf16=bf16)
        # tap_norm: same semantics as DINOv2Backbone (ablation knob)
        self.tap_norm = tap_norm
        cfg = DINOV3_CONFIGS[variant]

        if DINOV3_ROOT not in sys.path:
            sys.path.insert(0, DINOV3_ROOT)
        from dinov3.hub.backbones import dinov3_vitl16, Weights

        # Build architecture from the variant enum (SAT493M unties the
        # global/local cls norms), then load our converted state dict.
        weights_enum = getattr(Weights, cfg['weights'])
        self.encoder = dinov3_vitl16(pretrained=False, weights=weights_enum)
        if checkpoint:
            sd = torch.load(checkpoint, map_location='cpu')
            missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
            # rope periods + qkv bias masks are deterministic buffers the
            # model computes at init; everything else must match exactly.
            allowed = {m for m in missing
                       if 'rope_embed' in m or 'bias_mask' in m}
            assert not unexpected, f'unexpected keys: {unexpected[:5]}'
            assert set(missing) == allowed, f'missing keys: {missing[:5]}'

        L = cfg['num_layers']
        self.tap_indices = tap_indices or [
            L // 4 - 1, L // 2 - 1, 3 * L // 4 - 1, L - 1]

        self.adapter = ViTDetAdapter(cfg['embed_dim'], out_channels)

        self.register_buffer(
            'pixel_mean', torch.tensor(cfg['mean']).view(1, 3, 1, 1))
        self.register_buffer(
            'pixel_std', torch.tensor(cfg['std']).view(1, 3, 1, 1))

        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def forward(self, x):
        H, W = x.shape[-2:]
        x = (x - self.pixel_mean) / self.pixel_std

        with self.encoder_context():
            feats = self.encoder.get_intermediate_layers(
                x, n=self.tap_indices, reshape=True, norm=self.tap_norm)
        feats = [f.float() for f in feats]

        return self.adapter(feats, (H, W))
