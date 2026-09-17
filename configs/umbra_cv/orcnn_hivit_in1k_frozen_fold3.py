# Frozen ImageNet-MAE HiViT-B (SARATR-X's init, PRE-SAR) + Oriented R-CNN, fold 3.
# Single-variable control vs orcnn_saratrx_frozen_fold3: only the 186K-SAR
# continued-pretraining stage differs.
_base_ = ['./orcnn_saratrx_frozen_fold3.py']

model = dict(
    backbone=dict(
        checkpoint='weights/saratrx/mae_hivit_base_1600ep.pth'))
