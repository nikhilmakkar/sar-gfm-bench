# Frozen Copernicus-FM ViT-L/16 (spectral mode, X-band wavelength) + Oriented R-CNN, fold 1, bf16.
_base_ = ['./orcnn_dinov2l_frozen_fold1.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.CopernicusFMBackbone',
        checkpoint='weights/copernicusfm/CopernicusFM_ViT_large_varlang_e100.pth',
        frozen=True,
        bf16=True,
        out_channels=256))
