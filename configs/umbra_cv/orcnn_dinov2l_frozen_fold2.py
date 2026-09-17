# Frozen DINOv2 ViT-L/14 + Oriented R-CNN, Umbra CV fold 2.
# Dataset/eval come verbatim from the GrokSAR fold config (same folds and
# DotaBig2SmallMetric as every DenoDet run).
_base_ = [
    '../_base_/orcnn_gfm.py',
    '../_base_/schedules/frozen_12e.py',
    '../_base_/default_runtime.py',
    '../../../GrokSAR/configs/_base_/datasets/Umbra_512_CV_fold2.py',
]

model = dict(
    backbone=dict(
        type='mmdet.DINOv2Backbone',
        variant='vitl14',
        frozen=True,
        out_channels=256))

# Pipeline override: the inherited fold config uses bare type='Resize', which
# resolved to a rotated-box-capable class in the ORIGINAL env but resolves to
# mmcv.Resize (crashes on RotatedBoxes) since the env rebuild of 2026-05-21.
# Explicit mmdet.Resize is version-proof. Otherwise identical to the
# pipelines used by every DenoDet run.
_train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType',
         box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5,
         direction=['horizontal', 'vertical']),
    dict(type='mmrotate.RandomRotate', prob=0.5, angle_range=90),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]
_val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType',
         box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True,
         clip_object_border=False),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

# ViT-L forward is heavier than the ResNet-18 the fold config was tuned for
train_dataloader = dict(
    batch_size=16, num_workers=8, dataset=dict(pipeline=_train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=_val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=_val_pipeline))
