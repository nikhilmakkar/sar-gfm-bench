# Frozen Clay v1.5 ViT-L/8 (DINOv2-distilled EO GFM, SAR VV/VH) + Oriented R-CNN, fold 1, bf16.
_base_ = ['./orcnn_dinov2l_frozen_fold1.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.ClayBackbone',
        checkpoint='weights/clay/clay-v1.5.ckpt',
        frozen=True,
        bf16=True,
        out_channels=256))
