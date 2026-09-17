# Capella cross-sensor ZERO-SHOT evaluation (test-only).
# Model comes from an Umbra fold checkpoint via --checkpoint; the test set is
# the full 267-chip Capella tier. HBB metric (Capella GT has no angles).
import os

_base_ = ['../_base_/orcnn_gfm.py', '../_base_/default_runtime.py']

custom_imports = dict(
    imports=['backbones', 'groksar', 'evals'],
    allow_failed_imports=False)

dataset_type = 'groksar.DotaBig2SmallDataset'
data_root = os.environ.get('CAPELLA_DATA_ROOT', 'datasets/capella_test/')

METAINFO = {'classes': ('aircraft', ), 'palette': [(106, 0, 228)]}

test_pipeline = [
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

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_prefix=dict(img_path=data_root + 'images'),
        img_suffix='png',
        ann_file=data_root + 'annfiles/',
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True,
        pipeline=test_pipeline))
val_dataloader = test_dataloader

test_evaluator = dict(
    type='CapellaHBBMetric',
    metric='mAP',
    eval_mode='11points')
val_evaluator = test_evaluator

test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')
train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None
