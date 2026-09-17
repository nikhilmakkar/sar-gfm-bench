# SARATR-X HiViT-B FINETUNED (author-intended usage) + Oriented R-CNN, fold 3.
# Identical to the frozen probe except frozen=False; fp32 (grads through encoder);
# same AdamW schedule as every other run.
_base_ = ['./orcnn_saratrx_frozen_fold3.py']

model = dict(backbone=dict(frozen=False, bf16=False))
