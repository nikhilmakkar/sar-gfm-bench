"""DOFA v1 ViT-L/16 backbone wrapper (Group B — wavelength-conditioned GFM).

DOFA (Xiong et al. 2024, arXiv:2403.15356): MAE pretraining across five
sensors (S1 SAR, S2, Landsat, NAIP, EnMAP hyperspectral) with a hypernetwork
that generates patch-embed conv weights from per-channel wavelengths — the
direct ancestor of Copernicus-FM's spectral embed. Code vendored under
backbones/vendor/dofa (repo zhu-xlab/DOFA, two files, torch+timm only).

Umbra input convention:
- 1 channel, wavelength **3.75 um** — NOT a physical microwave wavelength:
  DOFA's pretraining assigned S1 GRD (VV,VH) the stand-in value [3.75, 3.75]
  (pretraining/datasets/waves.json), so 3.75 IS the model's learned "SAR
  channel" identity; a physical X-band value (~3.1e7 um-scale) would be far
  outside anything it ever saw. Contrast with Copernicus-FM, which encodes
  physical wavelengths via log-scaled Fourier expansion.
- z-score normalization with the same Umbra log-crop stats as Copernicus-FM
  (mean 137.88, std 72.39 on [0,255]).

The vendored OFAViT has no intermediate-tap API and a fixed 224-grid sin-cos
pos embed with no runtime resize, so this wrapper (a) bicubic-interpolates
the checkpoint's 14x14 grid to the model's img_size grid at load and
(b) runs the block loop itself to collect raw taps [5, 11, 17, 23].
Square inputs assumed (our pipeline is 512x512 everywhere).
"""

import math

import torch
import torch.nn.functional as F
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

UMBRA_LOG_MEAN = 137.88 / 255.0
UMBRA_LOG_STD = 72.39 / 255.0
SAR_WAVELENGTH_UM = 3.75   # DOFA's learned stand-in for S1 SAR channels


def _resize_pos_embed(pos_embed, new_grid):
    """(1, 1+g*g, D) checkpoint pos embed -> (1, 1+new*new, D)."""
    cls_tok, grid = pos_embed[:, :1], pos_embed[:, 1:]
    g = int(math.sqrt(grid.shape[1]))
    grid = grid.reshape(1, g, g, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_grid, new_grid),
                         mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
    return torch.cat([cls_tok, grid], dim=1)


@MODELS.register_module()
class DOFABackbone(BaseGFMBackbone):

    PATCH = 16

    def __init__(self, checkpoint=None, frozen=True, out_channels=256,
                 tap_indices=None, bf16=False, img_size=512):
        super().__init__(frozen=frozen, bf16=bf16)
        from backbones.vendor.dofa.dofa_v1 import vit_large_patch16
        self.tap_indices = tap_indices or [5, 11, 17, 23]
        self.encoder = vit_large_patch16(img_size=img_size, num_classes=0)
        if checkpoint:
            ck = torch.load(checkpoint, map_location='cpu')
            sd = ck.get('model', ck)
            if sd['pos_embed'].shape != self.encoder.pos_embed.shape:
                sd['pos_embed'] = _resize_pos_embed(
                    sd['pos_embed'], img_size // self.PATCH)
            missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
            ok = lambda k: k.startswith(('head', 'fc_norm', 'norm'))
            # mask_token/projector = MAE + DINOv2-distillation pretraining
            # heads, not part of the encoder
            ok_unexp = lambda k: ok(k) or k.startswith(('mask_token', 'projector'))
            assert not [k for k in missing if not ok(k)], missing[:5]
            assert not [k for k in unexpected if not ok_unexp(k)], unexpected[:5]

        self.adapter = ViTDetAdapter(1024, out_channels)

        self.register_buffer('pixel_mean', torch.tensor(UMBRA_LOG_MEAN))
        self.register_buffer('pixel_std', torch.tensor(UMBRA_LOG_STD))

        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def forward(self, x):
        H, W = x.shape[-2:]
        x = x.mean(dim=1, keepdim=True)
        x = (x - self.pixel_mean) / self.pixel_std
        enc = self.encoder
        with self.encoder_context():
            waves = torch.tensor([SAR_WAVELENGTH_UM], device=x.device).float()
            t, _ = enc.patch_embed(x, waves)
            t = t + enc.pos_embed[:, 1:, :]
            cls_tok = (enc.cls_token + enc.pos_embed[:, :1, :]).expand(
                t.shape[0], -1, -1)
            t = torch.cat([cls_tok, t], dim=1)
            hw = int(math.sqrt(t.shape[1] - 1))
            feats = []
            for i, blk in enumerate(enc.blocks):
                t = blk(t)
                if i in self.tap_indices:
                    f = t[:, 1:].reshape(t.shape[0], hw, hw, -1)
                    feats.append(f.permute(0, 3, 1, 2).contiguous())
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
