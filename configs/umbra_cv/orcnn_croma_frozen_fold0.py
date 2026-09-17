# Frozen CROMA-large SAR encoder (dedicated radar-optical contrastive ViT, 12-layer half-depth) + Oriented R-CNN, fold 0, bf16.
# Reduced batch: CROMA's attention has no flash/SDPA path (dense einsum QK^T
# + ALiBi bias), ~2.8GB/item at 512px/patch8 (4096 tokens) -> bs16 would need
# ~46GB alone. bs6 verified as a starting point once GPU is free; adjust if OOM.
_base_ = ['./orcnn_dinov2l_frozen_fold0.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.CromaBackbone',
        checkpoint='weights/croma/CROMA_large.pt',
        frozen=True,
        bf16=True,
        image_resolution=512,
        out_channels=256))

train_dataloader = dict(batch_size=6)
