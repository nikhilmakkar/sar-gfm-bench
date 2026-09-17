"""Plain ImageNet-MAE ViT-L/16 backbone wrapper (Group A baseline).

The 'standard ImageNet ViT' row of the benchmark: same architecture size
class as DINOv2/v3-L, same frozen probe, only the pretraining differs
(MAE on ImageNet-1k, Meta's mae_pretrain_vit_large.pth).

Built via timm (vit_large_patch16_224 at img_size=512 — timm resizes the
pos embed at load). Taps via timm's forward_intermediates with norm=False
(raw block outputs — the convention of MAE's own detection transfer,
ViTDet); tap_norm knob provided like the other ViT wrappers.
"""

import torch
from mmdet.registry import MODELS

from adapters.vitdet_adapter import ViTDetAdapter
from backbones.base_gfm import BaseGFMBackbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@MODELS.register_module()
class MAEViTBackbone(BaseGFMBackbone):

    PATCH = 16

    def __init__(self, checkpoint=None, frozen=True, out_channels=256,
                 tap_indices=None, bf16=False, tap_norm=False, img_size=512):
        super().__init__(frozen=frozen, bf16=bf16)
        import timm
        self.tap_norm = tap_norm
        self.encoder = timm.create_model(
            'vit_large_patch16_224', pretrained=False, img_size=img_size,
            num_classes=0)
        if checkpoint:
            ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
            sd = ck.get('model', ck)
            # timm's filter resizes the 14x14(+cls) pos embed to our grid
            from timm.models.vision_transformer import checkpoint_filter_fn
            sd = checkpoint_filter_fn(sd, self.encoder)
            missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
            ok = lambda k: k.startswith(('head', 'fc_norm', 'norm'))
            assert not [k for k in missing if not ok(k)], missing[:5]
            assert not [k for k in unexpected if not ok(k)], unexpected[:5]

        self.tap_indices = tap_indices or [5, 11, 17, 23]
        self.adapter = ViTDetAdapter(1024, out_channels)

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
        with self.encoder_context():
            feats = self.encoder.forward_intermediates(
                x, indices=self.tap_indices, norm=self.tap_norm,
                output_fmt='NCHW', intermediates_only=True)
        feats = [f.float() for f in feats]
        return self.adapter(feats, (H, W))
