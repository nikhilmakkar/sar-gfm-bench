"""HBB evaluation for the Capella cross-sensor tier.

Capella GT is horizontal-box only (no angle annotations exist), while our
detectors predict rotated boxes. Scoring rboxes against axis-aligned GT with
rotated IoU would punish correctly-oriented tight predictions on rotated
aircraft, so BOTH predictions and GT are converted to horizontal boxes
before AP computation. Inherits everything else (11-point mAP, class map)
from groksar's DotaBig2SmallMetric with predict_box_type='hbox'.
"""

import numpy as np
from mmdet.registry import METRICS

from groksar.evaluation.metrics.dota_big2small_metric import \
    DotaBig2SmallMetric


def rbox2hbox_np(rb: np.ndarray) -> np.ndarray:
    """(N,5) cx,cy,w,h,theta -> (N,4) x1,y1,x2,y2."""
    if rb.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h, t = rb[:, 0], rb[:, 1], rb[:, 2], rb[:, 3], rb[:, 4]
    cos, sin = np.abs(np.cos(t)), np.abs(np.sin(t))
    dw = (w * cos + h * sin) / 2
    dh = (w * sin + h * cos) / 2
    return np.stack([cx - dw, cy - dh, cx + dw, cy + dh], axis=1)


@METRICS.register_module()
class CapellaHBBMetric(DotaBig2SmallMetric):

    def __init__(self, **kwargs):
        kwargs.setdefault('predict_box_type', 'hbox')
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples):
        for ds in data_samples:
            for key in ('gt_instances', 'ignored_instances'):
                inst = ds.get(key)
                if inst and 'bboxes' in inst and inst['bboxes'].numel():
                    if inst['bboxes'].shape[-1] == 5:
                        import torch
                        hb = rbox2hbox_np(inst['bboxes'].cpu().numpy())
                        inst['bboxes'] = torch.from_numpy(hb)
            pred = ds['pred_instances']
            if pred['bboxes'].numel() and pred['bboxes'].shape[-1] == 5:
                import torch
                hb = rbox2hbox_np(pred['bboxes'].cpu().numpy())
                pred['bboxes'] = torch.from_numpy(hb)
            else:
                import torch
                pred['bboxes'] = torch.zeros((0, 4))
        super().process(data_batch, data_samples)
