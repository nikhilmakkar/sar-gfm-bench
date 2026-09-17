# Frozen OlmoEarth-v1-Large (Ai2 multimodal EO GFM, SAR-only input, true ViT-L) + Oriented R-CNN, fold 0, bf16.
_base_ = ['./orcnn_dinov2l_frozen_fold0.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.OlmoEarthBackbone',
        checkpoint='weights/olmoearth_large/weights.pth',
        config_path='weights/olmoearth_large/config.json',
        frozen=True,
        bf16=True,
        out_channels=256))
