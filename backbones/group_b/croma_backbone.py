"""CROMA-large backbone wrapper (Group B — genuinely SAR-native contrastive GFM).

CROMA (Fuller et al., NeurIPS 2023): contrastive radar-optical masked
autoencoding on paired Sentinel-1/Sentinel-2. Unlike Copernicus-FM/DOFA/Clay,
which treat SAR as one of MANY generic modalities via a wavelength
hypernetwork, CROMA has a DEDICATED, purpose-built SAR encoder (2-channel
VV/VH ViT) trained contrastively against optical — the closest thing in this
benchmark to "designed specifically to understand SAR". Vendored verbatim
(backbones/vendor/croma/use_croma.py; single file, einops+torch only).

We run `modality='SAR'`: only the s1_encoder branch is built (no optical
encoder, no cross-attention fusion) — a clean SAR-only forward pass, no
"present/absent modality" masking convention to get right (unlike Galileo/
OlmoEarth). NOTE: CROMA's SAR branch is HALF the nominal encoder depth
(large: encoder_depth=24 total budget, but s1_encoder gets depth=12 — the
other 12 layers are spent on cross-modal fusion we don't use). Reported as
one more param/depth data point per the project's scale-is-reported-not-
gated ruling, not hidden.

Umbra input convention:
- CROMA wants 2 SAR channels (VV, VH); we duplicate our amplitude, same
  convention as DOFA/Clay/Galileo.
- Their own README normalizes via per-channel mean/std-clip to 8-bit then
  rescales to [0,1] ("taken from SatMAE and SeCo"); we instead z-score with
  Umbra's own log-crop stats (mean 137.88, std 72.39 on [0,255]) for
  consistency with every other wrapper in this benchmark.
- image_resolution passed as our actual crop size (512), not their 120px
  pretraining default — the model supports this (any multiple of 8) via
  their get_2dalibi position-bias, computed fresh for whatever num_patches
  the resolution implies.

MEMORY CAVEAT (documented, verify before scaling batch size): CROMA's
Attention has NO flash/SDPA path — it's raw einsum QK^T + dense ALiBi bias +
softmax, always materializing a (B, 16, num_patches, num_patches) tensor.
At 512px/patch8, num_patches=4096, so this is a ~1GB/head-tensor... a
~64x larger attention matrix than CROMA's native 120px (225 patches). This
is architecturally the most memory-hungry backbone in the benchmark; batch
size must be verified empirically (see queue config) rather than assumed.

Taps: raw block outputs at [3, 6, 9, 12] (quarter-depth fractions of the
12-layer SAR branch, matching the convention of the other backbones).
"""

import os
import sys

import torch
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vendor')
UMBRA_LOG_MEAN = 137.88 / 255.0
UMBRA_LOG_STD = 72.39 / 255.0
PATCH = 8


@MODELS.register_module()
class CromaBackbone(BaseGFMBackbone):

    def __init__(self, checkpoint=None, frozen=True, out_channels=256,
                 tap_indices=None, bf16=False, image_resolution=512):
        super().__init__(frozen=frozen, bf16=bf16)
        if VENDOR_DIR not in sys.path:
            sys.path.insert(0, VENDOR_DIR)
        from croma.use_croma import PretrainedCROMA
        # PretrainedCROMA.__init__ builds the module AND loads weights in one
        # call (no separate strict-check hook available) -- correctness is
        # instead verified by the smoke test's checkpoint-identity check.
        full = PretrainedCROMA(pretrained_path=checkpoint, size='large',
                               modality='SAR', image_resolution=image_resolution)
        self.encoder = full.s1_encoder     # ViT: linear_input + transformer
        self.attn_bias = full.attn_bias    # (1, 16, num_patches, num_patches), fixed geometry
        self.depth = self.encoder.depth    # 12 for large's SAR branch
        self.tap_indices = tap_indices or [
            max(0, self.depth // 4 - 1), self.depth // 2 - 1,
            (3 * self.depth) // 4 - 1, self.depth - 1]

        self.adapter = ViTDetAdapter(self.encoder.dim, out_channels)
        self.register_buffer('pixel_mean', torch.tensor(UMBRA_LOG_MEAN))
        self.register_buffer('pixel_std', torch.tensor(UMBRA_LOG_STD))

        if frozen:
            self.freeze_encoder(self.encoder)
            self.attn_bias.requires_grad_(False)
        self.log_params()

    def forward(self, x):
        H, W = x.shape[-2:]
        x1 = x.mean(dim=1, keepdim=True)
        x2 = torch.cat([x1, x1], dim=1)            # -> VV, VH
        x2 = (x2 - self.pixel_mean) / self.pixel_std
        side = H // PATCH
        with self.encoder_context():
            from einops import rearrange
            t = rearrange(x2, 'b c (h i) (w j) -> b (h w) (c i j)', i=PATCH, j=PATCH)
            t = self.encoder.linear_input(t)
            bias = self.attn_bias.to(x.device)
            feats = []
            for i, (attn, ffn) in enumerate(self.encoder.transformer.layers):
                t = attn(t, bias) + t
                t = ffn(t) + t
                if i in self.tap_indices:
                    f = t.reshape(t.shape[0], side, side, -1).permute(0, 3, 1, 2)
                    feats.append(f.contiguous())
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
