_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/gotham_track.py',
    '../_base_/default_runtime.py',
]

vis_interval = 1 # Visualize every image.
checkpoint_file = 'pretrain/swin_tiny_patch4_window7_224.pth'
warmup_steps = 1500
total_steps = 40000
val_interval = total_steps // 5
num_classes = 7

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='HeadCTVisualizer', vis_backends=vis_backends, name='visualizer')
data_preprocessor = dict(
      type='SegDataPreProcessor',
      size_divisor=32,  # We're using 256 for training, 512 for testing.
      mean=[124.95, 124.95, 124.95],
      std=[24.735, 24.735, 24.735],
)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=num_classes),
    auxiliary_head=dict(in_channels=384, num_classes=num_classes))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_steps),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=warmup_steps,
        end=total_steps,
        by_epoch=False,
    )
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='HeadCTVisualizationHook', interval=vis_interval))

# 20 per V100 GPU.
train_dataloader = dict(batch_size=20)
# 19636 (num images in TRACK) mod 4 = 0.
val_dataloader = dict(batch_size=4)
test_dataloader = val_dataloader

# Set evaluator.
val_evaluator = dict(type='IoUROCMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = val_evaluator

# Train/test configs.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=total_steps, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
