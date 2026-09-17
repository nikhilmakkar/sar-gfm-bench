import os

# SARATR-X finetuned, in-domain quadrant split.
_base_ = ['./orcnn_saratrx_ft_fold0.py']


dataset_type = 'groksar.DotaBig2SmallDataset'
backend_args = None
DATA_ROOT = os.environ.get(
    'UMBRA_INDOMAIN_ROOT', 'datasets/Umbra/umbra_indomain')

METAINFO = {
    'classes': ('aircraft',),
    'palette': [(106, 0, 228)]
}

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='mmrotate.RandomRotate', prob=0.5, angle_range=90),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True,
         clip_object_border=False),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_prefix=dict(img_path=f'{DATA_ROOT}/train/images'),
        ann_file=f'{DATA_ROOT}/train/annfiles/',
        img_suffix='png',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_prefix=dict(img_path=f'{DATA_ROOT}/val/images'),
        ann_file=f'{DATA_ROOT}/val/annfiles/',
        img_suffix='png',
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_prefix=dict(img_path=f'{DATA_ROOT}/test/images'),
        ann_file=f'{DATA_ROOT}/test/annfiles/',
        img_suffix='png',
        test_mode=True,
        pipeline=val_pipeline))

val_evaluator = dict(
    type='groksar.DotaBig2SmallMetric',
    metric='mAP',
    eval_mode='11points',
    predict_box_type='rbox')
test_evaluator = val_evaluator
