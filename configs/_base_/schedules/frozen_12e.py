# Schedule for frozen-encoder detection probing.
# AdamW on the trainable ~25M (adapter + FPN + heads); 12 epochs with the
# classic 1x decay points so runs stay comparable to the DenoDet schedule_1x.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True,
         milestones=[8, 11], gamma=0.1)
]
