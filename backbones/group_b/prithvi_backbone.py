"""Prithvi-EO-2.0-600M backbone wrapper (Group B — OPTICAL CONTROL, not SAR).

Prithvi-EO-2.0 (IBM/NASA, 2024): a video-ViT MAE pretrained on Sentinel-2
OPTICAL time series only (6 bands B02-B07, no SAR whatsoever). Included
deliberately as a wrong-modality control: if a pure-optical GFM lands near
the bottom of the frozen table (as expected), that validates the benchmark
is actually sensitive to modality match rather than just rewarding "any
large pretrained ViT". Largest available checkpoint (600M, ViT-H scale:
1280-dim, depth 32) per the project's params-are-reported-not-gated ruling.

Code vendored verbatim from the HF repo (backbones/vendor/prithvi/
prithvi_mae.py; single file, standard deps). We use PrithviViT.forward_features
directly — unlike Clay/Galileo/Copernicus/DOFA it needs NO custom tap-loop
hacking: forward_features already returns one tensor per block (no MAE
masking applied on this path) and prepare_features_for_image_model reshapes
tokens straight to NCHW.

Umbra input convention:
- The model expects 6 Sentinel-2 REFLECTANCE bands; we have 1 SAR amplitude
  channel with no principled reflectance mapping (this IS the point of a
  wrong-modality control) so we broadcast it identically into all 6 slots.
- z-scored with Umbra's own log-crop stats (mean 137.88, std 72.39 on
  [0,255]) rather than Prithvi's Sentinel-2 per-band stats, since our data
  has nothing to do with those units.
- num_frames=4 at pretrain, but patch_size[0]=1 and the model's own
  _interpolate_pos_encoding recomputes the sin-cos position embedding for
  any T (verified in source) -> we feed a single frame (T=1), matching
  every other backbone's single-image convention. No temporal/location
  coords needed (config coords_encoding=[] -> both disabled for this ckpt).

patch 14x14 -> 512/14 is not integer; PatchEmbed only warns and truncates
the border (see its own forward), which would silently crop ~4px asymmetric
-- we instead resize input to 518x518 (nearest multiple of 14 >= 512) so the
patch grid is exact, then the adapter resizes pyramid outputs back to (H,W)
of the ORIGINAL 512 input like every other wrapper.
"""

import os
import sys

import torch
import torch.nn.functional as F
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vendor')
UMBRA_LOG_MEAN = 137.88 / 255.0
UMBRA_LOG_STD = 72.39 / 255.0
PATCH = 14
RESIZED = 518   # 37 * 14, nearest multiple of 14 covering 512


@MODELS.register_module()
class PrithviBackbone(BaseGFMBackbone):

    def __init__(self, checkpoint=None, frozen=True, out_channels=256,
                 tap_indices=None, bf16=False):
        super().__init__(frozen=frozen, bf16=bf16)
        if VENDOR_DIR not in sys.path:
            sys.path.insert(0, VENDOR_DIR)
        from prithvi.prithvi_mae import PrithviViT
        self.tap_indices = tap_indices or [7, 15, 23, 31]  # /32, matches [5,11,17,23]/24 fraction
        # Build with the PRETRAINING shape (img_size=224, num_frames=4) so
        # pos_embed's buffer matches the checkpoint exactly (verified:
        # ckpt encoder.pos_embed is (1,1025,1280) = 4*16*16+1 cls, exactly
        # this ctor's grid) -> clean strict load. At forward time we feed a
        # single 518x518 frame; forward_features's own
        # interpolate_pos_encoding recomputes the sin-cos pos-embed for the
        # ACTUAL (T=1, 37x37) shape from the loaded (T=4, 16x16) buffer —
        # that adaptation is the documented purpose of that function, not a
        # workaround of ours. RESIZED/PATCH are the runtime shape only; the
        # ctor's own img_size/num_frames must stay at pretrain values.
        self.encoder = PrithviViT(
            img_size=224, patch_size=(1, PATCH, PATCH), num_frames=4,
            in_chans=6, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
            coords_encoding=[])
        if checkpoint:
            sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
            pfx = 'encoder.'
            enc_sd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
            missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
            assert not missing, missing[:5]
            assert not unexpected, unexpected[:5]

        self.adapter = ViTDetAdapter(1280, out_channels)
        self.register_buffer('pixel_mean', torch.tensor(UMBRA_LOG_MEAN))
        self.register_buffer('pixel_std', torch.tensor(UMBRA_LOG_STD))

        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def forward(self, x):
        H, W = x.shape[-2:]
        x1 = x.mean(dim=1, keepdim=True)              # 3 identical -> 1
        x1 = F.interpolate(x1, size=(RESIZED, RESIZED), mode='bilinear',
                           align_corners=False)
        x6 = x1.repeat(1, 6, 1, 1)                     # -> 6 pseudo-bands
        x6 = (x6 - self.pixel_mean) / self.pixel_std
        x6 = x6.unsqueeze(2)                            # [B,C,H,W] -> [B,C,T=1,H,W]
        with self.encoder_context():
            all_feats = self.encoder.forward_features(x6)
            taps = [all_feats[i] for i in self.tap_indices]
            # NOT encoder.prepare_features_for_image_model: it derives the
            # time-dim from the CTOR's static num_frames=4 (needed for
            # strict checkpoint loading, see __init__ comment), not the
            # T=1 we actually run -> wrong reshape divisor. We always run a
            # single frame, so the reshape is unambiguous: drop the cls
            # token, grid is square with side sqrt(num_tokens).
            feats = []
            for t in taps:
                tok = t[:, 1:, :]                       # drop cls
                side = int(tok.shape[1] ** 0.5)
                f = tok.reshape(tok.shape[0], side, side, -1).permute(0, 3, 1, 2)
                feats.append(f.contiguous())
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
