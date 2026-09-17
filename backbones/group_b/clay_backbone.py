"""Clay v1.5 ViT-L/8 backbone wrapper (Group B — DINOv2-distilled EO GFM).

Clay v1.5 (Clay Foundation, 2024): an MAE-style ViT-L pretrained on many EO
sensors (incl. Sentinel-1 SAR) and DISTILLED from a DINOv2-L teacher
(config teacher = vit_large_patch14_reg4_dinov2.lvd142m). That makes Clay a
direct test of "does EO-adapting our frozen winner (DINOv2) help or hurt?".
Like DOFA/Copernicus it has a wavelength hypernetwork (DynamicEmbedding) that
generates the patch conv from per-band wavelengths, plus a gsd/time/latlon
metadata position encoding.

Imported from the Clay repo at the Clay model repository (claymodel package;
model-level deps are timm/torchvision/einops only — no Lightning). We build
just the Encoder (mask_ratio=0, no masking, no shuffle) — never the full
ClayMAE, which would download the DINOv2 teacher and the decoder.

Umbra input convention:
- Clay S1-RTC expects 2 bands (VV, VH). Umbra is 1-channel amplitude, so we
  duplicate it into both slots and feed Clay's OWN S1 stand-in wavelengths
  (vv=3.5, vh=4.0 from configs/metadata.yaml — like DOFA's 3.75, these are
  NOT physical microwave values but the model's learned SAR channel identity).
- z-score with the Umbra log-crop stats (mean 137.88, std 72.39 on [0,255]),
  matching Copernicus-FM/DOFA. Clay's own S1 normalization is per-band dB
  z-score, so a standard-normal input is the faithful analogue.
- gsd=10 (Clay's S1 pretrain resolution) so the sincos+gsd position encoding
  stays in the regime it was trained on; Umbra's true gsd (~0.5 m) is far
  outside anything Clay saw. time/latlon = 0 (neutral). Both documented
  judgment calls, same family as the parked wavelength ablation.

patch_size=8 -> 64x64 token grid at 512 in; taps raw block outputs
[5,11,17,23] reshaped to that grid, adapter resizes to strides 8/16/32/64.
"""

import math
import os
import sys

import torch
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

CLAY_REPO = os.environ.get('CLAY_REPO', '../clay-model')
UMBRA_LOG_MEAN = 137.88 / 255.0
UMBRA_LOG_STD = 72.39 / 255.0
S1_WAVES = [3.5, 4.0]   # Clay's own VV/VH stand-in wavelengths
S1_GSD = 10.0           # Clay's S1 pretrain resolution (metadata.yaml)


@MODELS.register_module()
class ClayBackbone(BaseGFMBackbone):

    PATCH = 8

    def __init__(self, checkpoint=None, frozen=True, out_channels=256,
                 tap_indices=None, bf16=False):
        super().__init__(frozen=frozen, bf16=bf16)
        if CLAY_REPO not in sys.path:
            sys.path.insert(0, CLAY_REPO)
        from claymodel.model import Encoder
        self.tap_indices = tap_indices or [5, 11, 17, 23]
        self.encoder = Encoder(mask_ratio=0.0, patch_size=self.PATCH,
                               shuffle=False, dim=1024, depth=24, heads=16,
                               dim_head=64, mlp_ratio=4)
        # Clay keeps fused_attn=True (F.scaled_dot_product_attention). Patch-8
        # gives 4097 tokens, so the manual matmul path would materialize a
        # 4097x4097 attention matrix per head and OOM at batch 16; SDPA avoids
        # that. It needs nvrtc, which the queue script puts on LD_LIBRARY_PATH
        # (weights/../.libs/libnvrtc.so). Left as-is.
        if checkpoint:
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            sd = ck.get('state_dict', ck)
            pfx = 'model.encoder.'
            enc_sd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
            missing, unexpected = self.encoder.load_state_dict(
                enc_sd, strict=False)
            assert not missing, missing[:5]
            assert not unexpected, unexpected[:5]

        self.adapter = ViTDetAdapter(1024, out_channels)
        self.register_buffer('pixel_mean', torch.tensor(UMBRA_LOG_MEAN))
        self.register_buffer('pixel_std', torch.tensor(UMBRA_LOG_STD))

        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def forward(self, x):
        H, W = x.shape[-2:]
        x1 = x.mean(dim=1, keepdim=True)          # 3 identical channels -> 1
        x2 = torch.cat([x1, x1], dim=1)           # -> 2 bands (VV, VH)
        x2 = (x2 - self.pixel_mean) / self.pixel_std
        B = x2.shape[0]
        enc = self.encoder
        time = torch.zeros(B, 4, device=x.device)
        latlon = torch.zeros(B, 4, device=x.device)
        waves = torch.tensor(S1_WAVES, device=x.device).float()
        with self.encoder_context():
            patches, _ = enc.to_patch_embed(x2, waves)         # [B, L, 1024]
            gsd = torch.tensor(S1_GSD, device=x.device)
            patches = enc.add_encodings(patches, time, latlon, gsd)
            hw = int(math.sqrt(patches.shape[1]))
            cls = enc.cls_token.expand(B, -1, -1)
            t = torch.cat([cls, patches], dim=1)
            feats = []
            for i, (attn, ff) in enumerate(enc.transformer.layers):
                t = attn(t) + t
                t = ff(t) + t
                if i in self.tap_indices:
                    f = t[:, 1:].reshape(B, hw, hw, -1)
                    feats.append(f.permute(0, 3, 1, 2).contiguous())
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
