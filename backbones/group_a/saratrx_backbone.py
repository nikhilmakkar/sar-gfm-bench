"""SARATR-X backbone wrapper (Group A: HiViT-B, SAR-SSL pretrained).

The SAR-pretrained entry of the benchmark: HiViT-Base (depths [4,4,20],
main-stage dim 512, patch 16) MAE-pretrained on 186K SAR images
(SARATR-X, checkpoint-800 of weight/186K_all).

Input adaptation: NONE beyond the benchmark's [0,1] contract — SARATR-X
pretraining used torchvision ToTensor() with Normalize commented out
(codelab/code/pretrain/main_pretrain.py:130), i.e. raw [0,1] inputs.

Architecture handling: the pure-torch models_hivit.HiViT is imported by file
path (their mmdet fork would collide with ours). Forward mirrors their
detection fork's staged forward (hivit.py) minus windowed attention (32x32
tokens at 512^2 input is cheap to attend globally) and minus rpe (the MAE
checkpoint carries no relative-position tables; rpe=False at build so the
frozen encoder holds no random untrained parameters).

Taps: main-stage blocks [4, 9, 14, 19] of 20 (the det fork's own default,
= L/4 spacing) -> four (B, 512, H/16, W/16) maps -> ViTDetAdapter.
"""

import importlib.util
import math
import os

import torch
import torch.nn.functional as F
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

SARATRX_ROOT = os.environ.get('SARATRX_ROOT', '../SARATR-X')
HIVIT_PY = os.path.join(
    SARATRX_ROOT, 'codelab/code/detection/models/models_hivit.py')


def _load_hivit_module():
    spec = importlib.util.spec_from_file_location('saratrx_models_hivit',
                                                  HIVIT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@MODELS.register_module()
class SARATRXBackbone(BaseGFMBackbone):

    PATCH = 16

    def __init__(self, checkpoint=None, frozen=True, out_channels=256,
                 tap_indices=None, bf16=False, img_size=512):
        super().__init__(frozen=frozen, bf16=bf16)
        mod = _load_hivit_module()
        # depths [2,2,20]: this constructor builds 2*d early blocks per stage,
        # which is what checkpoint-800's layout (4+4+20 blocks) corresponds to
        self.encoder = mod.HiViT(
            img_size=img_size, patch_size=16, embed_dim=512,
            depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
            ape=True, rpe=False, num_classes=0)
        self.tap_indices = tap_indices or [4, 9, 14, 19]

        if checkpoint:
            self._load_mae_checkpoint(checkpoint)

        self.adapter = ViTDetAdapter(512, out_channels)

        if frozen:
            self.freeze_encoder(self.encoder)
        self.log_params()

    def _load_mae_checkpoint(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        sd = ck.get('model', ck)
        # drop MAE decoder, mask token, and the HOG pretraining-target
        # filters (hogs*) — all loss-side machinery, not encoder weights
        sd = {k: v for k, v in sd.items()
              if not k.startswith(('decoder', 'hogs')) and k != 'mask_token'}
        # pretraining pos embed is 14x14 tokens (224^2); resize to ours
        pe = sd.get('absolute_pos_embed')
        tgt = self.encoder.absolute_pos_embed
        if pe is not None and pe.shape[1] != tgt.shape[1]:
            n_src = int(math.sqrt(pe.shape[1]))
            n_tgt = int(math.sqrt(tgt.shape[1]))
            pe2 = pe.reshape(1, n_src, n_src, -1).permute(0, 3, 1, 2)
            pe2 = F.interpolate(pe2, size=(n_tgt, n_tgt), mode='bicubic',
                                align_corners=False)
            sd['absolute_pos_embed'] = pe2.permute(0, 2, 3, 1).reshape(
                1, n_tgt * n_tgt, -1)
        missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
        # Allowed mismatches, all unused by our tap-based forward:
        #   head.* / fc_norm.* — classifier head (num_classes=0 here);
        #   norm.* — MAE names the final norm 'norm', this class 'fc_norm';
        #   neither is applied to intermediate block outputs.
        ok = lambda k: k.startswith(('head', 'fc_norm', 'norm'))
        bad_missing = [k for k in missing if not ok(k)]
        bad_unexpected = [k for k in unexpected if not ok(k)]
        assert not bad_missing, f'missing: {bad_missing[:5]}'
        assert not bad_unexpected, f'unexpected: {bad_unexpected[:5]}'

    def forward(self, x):
        H, W = x.shape[-2:]
        # no normalization: SARATR-X pretraining consumed raw [0,1] pixels
        enc = self.encoder
        with self.encoder_context():
            feats = []
            B = x.shape[0]
            Hp, Wp = H // self.PATCH, W // self.PATCH
            t = enc.patch_embed(x)
            for blk in enc.blocks[:-enc.num_main_blocks]:
                t = blk(t)
            t = t[..., 0, 0, :]
            pe = enc.absolute_pos_embed
            if pe.shape[1] != t.shape[1]:
                n_src = int(math.sqrt(pe.shape[1]))
                pe = F.interpolate(
                    pe.reshape(1, n_src, n_src, -1).permute(0, 3, 1, 2),
                    size=(Hp, Wp), mode='bicubic', align_corners=False
                ).permute(0, 2, 3, 1).reshape(1, Hp * Wp, -1)
            t = t + pe
            t = enc.pos_drop(t)
            for i, blk in enumerate(enc.blocks[-enc.num_main_blocks:]):
                t = blk(t)
                if i in self.tap_indices:
                    feats.append(t.permute(0, 2, 1).reshape(
                        B, -1, Hp, Wp).contiguous())
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
