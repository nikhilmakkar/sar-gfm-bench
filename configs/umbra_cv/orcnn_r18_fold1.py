# ResNet-18 (fully trained, torchvision init) + Oriented R-CNN, fold 1.
# The continuity baseline: same backbone family as the DenoDet runs, but
# inside the new fixed instrument (plain FPN + Oriented R-CNN), so old and
# new numbers connect. Uses stock ImageNet preprocessing since the backbone
# trains end-to-end.
_base_ = [
    '../_base_/orcnn_gfm.py',
    '../_base_/schedules/frozen_12e.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/umbra_cv_fold1.py',
]

model = dict(
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    backbone=dict(
        _delete_=True,
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]),
    rpn_head=dict(
        anchor_generator=dict(strides=[4, 8, 16, 32, 64])),
    roi_head=dict(
        bbox_roi_extractor=dict(featmap_strides=[4, 8, 16, 32])))

# Pipeline override: explicit mmdet.Resize — see orcnn_dinov2l_frozen_fold1.py
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

train_dataloader = dict(
    batch_size=32, num_workers=8, dataset=dict(pipeline=_train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=_val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=_val_pipeline))
