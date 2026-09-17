# Frozen DINOv3 ViT-L/16 (sat) + Oriented R-CNN, Umbra CV fold 1, bf16.
# Everything except the backbone is inherited from the DINOv2-L fold config
# (same head, schedule, data, eval). Weights are the license-gated HF
# checkpoint, downloaded to weights/dinov3/ (see labbook).
_base_ = ['./orcnn_dinov2l_frozen_fold1.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.DINOv3Backbone',
        variant='vitl16_sat',
        checkpoint='weights/dinov3/dinov3_vitl16_sat.pth',
        frozen=True,
        bf16=True,
        out_channels=256))
