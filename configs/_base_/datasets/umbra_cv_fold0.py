"""Portable DOTA aircraft dataset config for cross-acquisition fold 0."""
import os

fold = 0
data_root = os.path.abspath(os.environ.get(
    'UMBRA_CV_ROOT', 'datasets/Umbra/umbra_cv'))
fold_root = os.path.join(data_root, f'fold_{fold}')

dataset_type = 'groksar.DotaBig2SmallDataset'
backend_args = None
metainfo = {'classes': ('aircraft',), 'palette': [(106, 0, 228)]}

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType',
         box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5,
         direction=['horizontal', 'vertical']),
    dict(type='mmrotate.RandomRotate', prob=0.5, angle_range=90),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType',
         box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True,
         clip_object_border=False),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_prefix=dict(img_path=os.path.join(fold_root, 'train/images')),
        img_suffix='png',
        ann_file=os.path.join(fold_root, 'train/annfiles') + os.sep,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_prefix=dict(img_path=os.path.join(fold_root, 'val/images')),
        img_suffix='png',
        ann_file=os.path.join(fold_root, 'val/annfiles') + os.sep,
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(
    type='groksar.DotaBig2SmallMetric',
    metric='mAP',
    eval_mode='11points',
    predict_box_type='rbox')
test_evaluator = val_evaluator
