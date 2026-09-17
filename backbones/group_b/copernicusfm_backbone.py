"""Copernicus-FM ViT-L/16 backbone wrapper (Group B — real GFM #1).

Copernicus-FM (Wang et al. 2025, arXiv:2503.11849): MAE-style pretraining on
Copernicus-Pretrain (S1 GRD / S2 TOA / S3 OLCI / S5P / DEM), with a spectral
hypernetwork that GENERATES the patch-embed conv weights from each channel's
central wavelength + bandwidth — so arbitrary sensors are first-class inputs.
Code vendored under backbones/vendor/copernicusfm (repo zhu-xlab/Copernicus-FM,
standalone PyTorch + timm Blocks; no terratorch needed).

Umbra input convention (spectral mode, mirroring their S1 GRD recipe):
- 1 channel (our 3 channels are identical replicas of the log-scaled X-band
  amplitude; we average them back to 1). Their S1 uses 2 channels (VV, VH)
  with wavelength 5e7 nm (C-band ~5 cm) and bandwidth 1e9 nm (placeholder).
  Umbra is X-band: wavelength 3.1e7 nm (~3.1 cm), same bandwidth placeholder.
  Both are inside the hypernetwork's FourierExpansion range (100 nm .. 1e9 nm).
- z-score normalization (their stated recommendation; their S1 pretrain used
  per-channel dB z-score). Stats computed over 400 umbra_indomain train crops
  of the log-scaled 8-bit data: mean 137.88, std 72.39 (on [0,255]).
- metadata (lon/lat/time/area) = NaN -> the model's learned fallback tokens
  (the author-supported "meta info not available" path). Our crops' geo/time
  are knowable, but the acquisitions' GSD (~0.35-0.5 m) is far outside the
  pretraining GSDs (>=10 m) anyway; the learned tokens are the safer default.

Taps: raw block outputs [5, 11, 17, 23] via the model's own
return_intermediate/intermediate_indices (already NCHW) — same depth
fractions as the other ViT-L rows, raw-tap convention (ablation-verified
immaterial). Requires square inputs (their token grid is sqrt-derived);
our pipeline is 512x512 everywhere.
"""

import torch
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

UMBRA_LOG_MEAN = 137.88 / 255.0   # z-score stats of log-scaled train crops,
UMBRA_LOG_STD = 72.39 / 255.0     # rescaled to the [0,1] preprocessor output
XBAND_WAVELENGTH_NM = 3.1e7       # Umbra X-band ~9.7 GHz -> lambda ~3.1 cm
XBAND_BANDWIDTH_NM = 1e9          # same placeholder the authors use for S1


@MODELS.register_module()
class CopernicusFMBackbone(BaseGFMBackbone):

    PATCH = 16

    def __init__(self, checkpoint=None, frozen=True, out_channels=256,
                 tap_indices=None, bf16=False):
        super().__init__(frozen=frozen, bf16=bf16)
        from backbones.vendor.copernicusfm.model_vit import vit_large_patch16
        self.tap_indices = tap_indices or [5, 11, 17, 23]
        # img_size=224 matches the checkpoint's 14x14(+cls) sin-cos pos embed;
        # forward_features resizes it to the actual token grid every call.
        self.encoder = vit_large_patch16(
            img_size=224, num_classes=0, global_pool=True,
            return_intermediate=True, intermediate_indices=self.tap_indices)
        if checkpoint:
            ck = torch.load(checkpoint, map_location='cpu')
            sd = ck.get('model', ck)
            missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
            ok = lambda k: k.startswith(('head', 'fc_norm', 'norm'))
            assert not [k for k in missing if not ok(k)], missing[:5]
            assert not unexpected, unexpected[:5]

        self.adapter = ViTDetAdapter(1024, out_channels)

        self.register_buffer('pixel_mean', torch.tensor(UMBRA_LOG_MEAN))
        self.register_buffer('pixel_std', torch.tensor(UMBRA_LOG_STD))

        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def forward(self, x):
        H, W = x.shape[-2:]
        x = x.mean(dim=1, keepdim=True)          # 3 identical channels -> 1
        x = (x - self.pixel_mean) / self.pixel_std
        meta = torch.full((x.shape[0], 4), float('nan'), device=x.device)
        with self.encoder_context():
            _, feats = self.encoder.forward_features(
                x, meta, [XBAND_WAVELENGTH_NM], [XBAND_BANDWIDTH_NM],
                None, 'spectral', kernel_size=self.PATCH)
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
