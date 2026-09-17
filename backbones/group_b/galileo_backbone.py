"""Galileo-base backbone wrapper (Group B — multimodal space-time-band GFM).

Galileo (Tseng et al. 2025, nasaharvest/galileo): a multimodal EO transformer
pretrained with masked modelling over MANY modalities at once — Sentinel-1
SAR, Sentinel-2 optical, ERA5 weather, SRTM elevation, land cover, etc. Its
input is one structured tensor with a slot per modality plus a MASK marking
which slots are present; the encoder only attends to unmasked (present) slots.
Vendored via the repo's own single_file_galileo.py (designed for porting).

We feed SAR ONLY: fill the Sentinel-1 (VV/VH) slot, mask every other modality
absent — so it runs as a SAR-only encoder. Convention taken verbatim from the
repo's construct_galileo_input (mask 1 = absent, 0 = present; providing S1
flips only the S1 group to 0).

Umbra input convention (same family as the other GFMs):
- Umbra is 1-channel X-band amplitude; Galileo's SAR slot is Sentinel-1 C-band
  VV/VH (2 bands). We duplicate the amplitude into both VV and VH.
- z-score with Umbra log-crop stats (mean 137.88, std 72.39 on [0,255]);
  Galileo's own S1 norm is dB z-score, so standard-normal is the analogue.
- months = 5 (neutral), latlon/weather/elevation all masked absent.

CAVEATS specific to Galileo (documented, not hidden):
- Scale: Galileo-base is ViT-B (768-d, depth 12), not ViT-L. Per the project
  ruling, params are a reported column, not a fairness gate.
- SPATIAL out-of-distribution: Galileo trained on 4-12 patch grids; our
  512px @ patch-8 is a 64x64 grid, far larger than anything it saw. Positional
  encoding is computed for the actual size so it runs, but this is OOD.

The encoder is single-exit, so for the 4-level FPN we call forward four times
with exit_after in {3,6,9,12} and take the Sentinel-1 group's spatial tokens
(t=0) at each depth -> [B, H/8, W/8, 768] -> NCHW; adapter resizes to strides.
"""

import os
import sys
from pathlib import Path

import torch
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

GALILEO_REPO = os.environ.get('GALILEO_REPO', '../galileo')
UMBRA_LOG_MEAN = 137.88 / 255.0
UMBRA_LOG_STD = 72.39 / 255.0
PATCH = 8
DEPTH = 12
TAPS = [3, 6, 9, 12]
S1_GROUP_IDX = 0
DEFAULT_MONTH = 5
BASE_GSD = 10
EMBED_DIM = 768


@MODELS.register_module()
class GalileoBackbone(BaseGFMBackbone):

    def __init__(self, checkpoint=None, frozen=True, out_channels=256,
                 bf16=False):
        super().__init__(frozen=frozen, bf16=bf16)
        if GALILEO_REPO not in sys.path:
            sys.path.insert(0, GALILEO_REPO)
        import single_file_galileo as G
        self.G = G
        self.encoder = G.Encoder.load_from_folder(
            Path(checkpoint), torch.device('cpu'))
        # Our S1-only feed leaves zero masked tokens after Galileo's
        # remove_masked_tokens, so every block receives an ALL-TRUE attn_mask
        # (verified by hook) — a mathematical no-op that still forces the
        # masked-SDPA path and materializes a [B,heads,N,N] mask per block:
        # OOM at batch 16 (33GB+). Nullify provably-no-op masks so SDPA takes
        # the unmasked fast path; identical math (all-True mask == no mask).
        def _nullify_noop_mask(attn):
            orig = attn.forward
            def fwd(x, y=None, attn_mask=None):
                if attn_mask is not None and bool(attn_mask.all()):
                    attn_mask = None
                return orig(x, y=y, attn_mask=attn_mask)
            return fwd
        for blk in self.encoder.blocks:
            blk.attn.forward = _nullify_noop_mask(blk.attn)

        self.adapter = ViTDetAdapter(EMBED_DIM, out_channels)
        self.register_buffer('pixel_mean', torch.tensor(UMBRA_LOG_MEAN))
        self.register_buffer('pixel_std', torch.tensor(UMBRA_LOG_STD))
        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def _build_input(self, x):
        """x: [B, 2, H, W] normalized VV/VH -> Galileo's masked modality dict."""
        G = self.G
        B, _, H, W = x.shape
        dev = x.device
        n_st, n_stg = len(G.SPACE_TIME_BANDS), len(G.SPACE_TIME_BANDS_GROUPS_IDX)
        n_sp, n_spg = len(G.SPACE_BANDS), len(G.SPACE_BAND_GROUPS_IDX)
        n_t, n_tg = len(G.TIME_BANDS), len(G.TIME_BAND_GROUPS_IDX)
        n_stat, n_statg = len(G.STATIC_BANDS), len(G.STATIC_BAND_GROUPS_IDX)

        s_t_x = torch.zeros(B, H, W, 1, n_st, device=dev)
        s_t_x[:, :, :, 0, 0] = x[:, 0]        # VV
        s_t_x[:, :, :, 0, 1] = x[:, 1]        # VH
        s_t_m = torch.ones(B, H, W, 1, n_stg, device=dev)
        s_t_m[:, :, :, :, S1_GROUP_IDX] = 0   # S1 present, all else absent

        sp_x = torch.zeros(B, H, W, n_sp, device=dev)
        sp_m = torch.ones(B, H, W, n_spg, device=dev)
        t_x = torch.zeros(B, 1, n_t, device=dev)
        t_m = torch.ones(B, 1, n_tg, device=dev)
        st_x = torch.zeros(B, n_stat, device=dev)
        st_m = torch.ones(B, n_statg, device=dev)
        months = torch.full((B, 1), DEFAULT_MONTH, dtype=torch.long, device=dev)
        return s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months

    def forward(self, x):
        H, W = x.shape[-2:]
        x1 = x.mean(dim=1, keepdim=True)          # 3 identical channels -> 1
        x2 = torch.cat([x1, x1], dim=1)           # -> VV, VH
        x2 = (x2 - self.pixel_mean) / self.pixel_std
        inp = self._build_input(x2)
        feats = []
        with self.encoder_context():
            for k in TAPS:
                out = self.encoder(*inp, patch_size=PATCH,
                                   input_resolution_m=BASE_GSD, exit_after=k,
                                   add_layernorm_on_exit=True)
                s_t = out[0]                       # [B, H', W', T, C_g, D]
                g = s_t[:, :, :, 0, S1_GROUP_IDX, :]   # [B, H', W', D]
                feats.append(g.permute(0, 3, 1, 2).contiguous())
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
