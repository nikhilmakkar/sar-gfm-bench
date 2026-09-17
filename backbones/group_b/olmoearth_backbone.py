"""OlmoEarth-v1-Large backbone wrapper (Group B — Ai2 multimodal EO GFM).

OlmoEarth (Ai2, 2025): a FlexiViT-style multimodal masked-modelling encoder
over Sentinel-1/2, Landsat, etc. True ViT-L (1024-d, depth 24, patch 8).
Imported from the repo at the OlmoEarth repository (the `helios`/
`olmoearth_pretrain` package). We build ONLY the Encoder via EncoderConfig
(never the full LatentMIM train wrapper, which pulls in FSDP2 training APIs).

Getting it to load/run under this env (torch 2.0.1+cu118, py3.10) needed four
compatibility shims — all applied here before the import, all touching only
training-time paths never hit in a frozen forward:
  1. stub torch.distributed.fsdp.fully_shard / register_fsdp_forward_method
     (FSDP2 APIs absent in torch 2.0.1; only called inside apply_fsdp()).
  2. stub torch.distributed.DeviceMesh / .tensor.distribute_tensor
     (only used as type annotations / a training-only method).
  3. backport enum.StrEnum (py3.11-only; stdlib-compatible shim).
  4. `import torch._dynamo` so attention.py's @torch._dynamo.disable() resolves.
Plus one real vendor bug-fix committed in the repo's flexi_vit.py:
remove_masked_tokens' torch.sort doesn't support bool dtype on CUDA in
torch 2.0.1 -> upcast-to-uint8-then-back (numerically identical).

Input convention (SAR-only, like Galileo):
- OlmoEarth wants channels-LAST (B,H,W,T,C). Our 1-channel amplitude is
  duplicated into Sentinel-1's 2 bands (VV,VH) -> (B,H,W,1,2).
- the per-modality MASK must be (B,H,W,T,n_bandsets=1) filled with
  MaskValue.ONLINE_ENCODER(=0) = "present, seen by encoder". NOTE: do NOT
  use MaskedOlmoEarthSample.from_olmoearthsample — that helper builds the
  mask from the per-INSTANCE (unbatched) shape and silently drops the batch
  dim for a batched sample, mangling everything downstream. Build the mask
  explicitly.
- z-score with Umbra log-crop stats (mean 137.88 std 72.39 on [0,255]);
  timestamps a fixed neutral [15,6,2023]; input_res=10 (S1 pretrain GSD,
  vs Umbra's true ~0.5m — same documented judgment call as Clay/Galileo).

Single-exit model -> 4-level FPN via 4 forward calls with
token_exit_cfg={'sentinel1': k} for k in {6,12,18,24}; take the S1-group
spatial tokens (t=0) at each depth -> [B,H/8,W/8,1024] -> NCHW.
"""

import os
import sys

import torch
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

OLMO_REPO = os.environ.get('OLMO_REPO', '../olmoearth_pretrain')
CONFIG_PATH = os.environ.get(
    'OLMO_CONFIG_PATH', 'weights/olmoearth_large/config.json')
UMBRA_LOG_MEAN = 137.88 / 255.0
UMBRA_LOG_STD = 72.39 / 255.0
PATCH = 8
S1_GSD = 10
TAPS = [6, 12, 18, 24]
EMBED_DIM = 1024


def _apply_env_shims():
    import enum
    if not hasattr(enum, 'StrEnum'):
        class StrEnum(str, enum.Enum):
            def __str__(self):
                return str(self.value)
        enum.StrEnum = StrEnum
    import torch._dynamo  # noqa: F401  (attention.py uses torch._dynamo.disable())
    import torch.distributed as _dist
    import torch.distributed.fsdp as _fsdp
    _training_only = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError('training-only FSDP2 path, not used for frozen inference'))
    if not hasattr(_fsdp, 'fully_shard'):
        _fsdp.fully_shard = _training_only
    if not hasattr(_fsdp, 'register_fsdp_forward_method'):
        _fsdp.register_fsdp_forward_method = lambda *a, **k: None
    if not hasattr(_dist, 'DeviceMesh'):
        _dist.DeviceMesh = type('DeviceMesh', (), {})
    import torch.distributed.tensor as _dtensor
    if not hasattr(_dtensor, 'distribute_tensor'):
        _dtensor.distribute_tensor = _training_only


@MODELS.register_module()
class OlmoEarthBackbone(BaseGFMBackbone):

    def __init__(self, checkpoint=None, config_path=CONFIG_PATH, frozen=True,
                 out_channels=256, bf16=False):
        super().__init__(frozen=frozen, bf16=bf16)
        _apply_env_shims()
        if OLMO_REPO not in sys.path:
            sys.path.insert(0, OLMO_REPO)
        import json
        from olmoearth_pretrain.model_loader import patch_legacy_encoder_config
        import olmoearth_pretrain.nn.flexi_vit as fv
        from olmoearth_pretrain.datatypes import MaskValue
        self._MaskedSample = __import__(
            'olmoearth_pretrain.datatypes', fromlist=['MaskedOlmoEarthSample']
        ).MaskedOlmoEarthSample
        self._present = float(MaskValue.ONLINE_ENCODER.value)

        cfg = patch_legacy_encoder_config(json.load(open(config_path)))
        self.encoder = fv.EncoderConfig.from_dict(
            cfg['model']['encoder_config']).build()
        if checkpoint:
            sd = torch.load(checkpoint, map_location='cpu')
            enc_sd = {k[len('encoder.'):]: v for k, v in sd.items()
                      if k.startswith('encoder.')}
            missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
            assert not missing, missing[:5]
            assert not unexpected, unexpected[:5]

        self.adapter = ViTDetAdapter(EMBED_DIM, out_channels)
        self.register_buffer('pixel_mean', torch.tensor(UMBRA_LOG_MEAN))
        self.register_buffer('pixel_std', torch.tensor(UMBRA_LOG_STD))
        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def forward(self, x):
        H, W = x.shape[-2:]
        B = x.shape[0]
        x1 = x.mean(dim=1)                              # (B,H,W), 1 channel
        x1 = (x1 - self.pixel_mean) / self.pixel_std
        s1 = torch.stack([x1, x1], dim=-1).unsqueeze(3)  # (B,H,W,T=1,C=2) VV,VH
        mask = torch.full((B, H, W, 1, 1), self._present, device=x.device)
        ts = torch.tensor([[15, 6, 2023]], device=x.device).repeat(B, 1).unsqueeze(1)
        sample = self._MaskedSample(timestamps=ts, sentinel1=s1, sentinel1_mask=mask)
        feats = []
        with self.encoder_context():
            for k in TAPS:
                out = self.encoder(sample, patch_size=PATCH, input_res=S1_GSD,
                                   token_exit_cfg={'sentinel1': k})
                g = out['tokens_and_masks'].sentinel1[:, :, :, 0, 0, :]  # (B,Hp,Wp,D)
                feats.append(g.permute(0, 3, 1, 2).contiguous())
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
