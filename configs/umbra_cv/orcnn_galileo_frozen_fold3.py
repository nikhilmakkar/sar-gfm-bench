# Frozen Galileo-base (multimodal EO GFM, SAR-only input) + Oriented R-CNN, fold 3, bf16.
_base_ = ['./orcnn_dinov2l_frozen_fold3.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.GalileoBackbone',
        checkpoint='weights/galileo_base',
        frozen=True,
        bf16=True,
        out_channels=256))
