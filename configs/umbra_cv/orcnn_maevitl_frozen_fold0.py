# Frozen ImageNet-MAE ViT-L/16 (plain ViT baseline) + Oriented R-CNN, fold 0, bf16.
_base_ = ['./orcnn_dinov2l_frozen_fold0.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.MAEViTBackbone',
        checkpoint='weights/mae_pretrain_vit_large.pth',
        frozen=True,
        bf16=True,
        out_channels=256))
