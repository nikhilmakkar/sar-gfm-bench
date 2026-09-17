# Frozen ImageNet-MAE HiViT-B (SARATR-X's init, PRE-SAR) + Oriented R-CNN, fold 2.
# Single-variable control vs orcnn_saratrx_frozen_fold2: only the 186K-SAR
# continued-pretraining stage differs.
_base_ = ['./orcnn_saratrx_frozen_fold2.py']

model = dict(
    backbone=dict(
        checkpoint='weights/saratrx/mae_hivit_base_1600ep.pth'))
