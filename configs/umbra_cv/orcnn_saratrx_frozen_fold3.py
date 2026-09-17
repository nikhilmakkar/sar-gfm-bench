# Frozen SARATR-X HiViT-B (SAR-SSL, 186K images) + Oriented R-CNN, fold 3, bf16.
_base_ = ['./orcnn_dinov2l_frozen_fold3.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.SARATRXBackbone',
        checkpoint='weights/saratrx/checkpoint-800.pth',
        frozen=True,
        bf16=True,
        out_channels=256))
