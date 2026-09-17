# ABLATION: DINOv2-L fold 0 with RAW taps (norm=False — Meta's dense-task
# convention) vs the benchmark run's normed taps (their API default).
# fp32, identical to orcnn_dinov2l_frozen_fold0 in every other respect.
# Reference (normed taps): best AP50 0.575.
_base_ = ['./orcnn_dinov2l_frozen_fold0.py']

model = dict(backbone=dict(tap_norm=False))
