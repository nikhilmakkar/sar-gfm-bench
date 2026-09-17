# Frozen Prithvi-EO-2.0-600M (optical-only control, wrong modality) + Oriented R-CNN, fold 3, bf16.
_base_ = ['./orcnn_dinov2l_frozen_fold3.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.PrithviBackbone',
        checkpoint='weights/prithvi/Prithvi_EO_V2_600M.pt',
        frozen=True,
        bf16=True,
        out_channels=256))
