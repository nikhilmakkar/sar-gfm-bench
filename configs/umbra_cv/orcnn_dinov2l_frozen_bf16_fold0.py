# bf16 twin of orcnn_dinov2l_frozen_fold0: identical in every respect except
# the encoder forward runs under bfloat16 autocast. Reference fp32 result:
# best AP50 0.575. Adopt bf16 for the sweep iff this matches within noise.
_base_ = ['./orcnn_dinov2l_frozen_fold0.py']

model = dict(backbone=dict(bf16=True))
