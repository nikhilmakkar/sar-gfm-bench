# Frozen DOFA v1 ViT-L/16 (wavelength-conditioned, SAR channel 3.75um) + Oriented R-CNN, fold 0, bf16.
_base_ = ['./orcnn_dinov2l_frozen_fold0.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.DOFABackbone',
        checkpoint='weights/dofa/DOFA_ViT_large_e100.pth',
        frozen=True,
        bf16=True,
        out_channels=256))
