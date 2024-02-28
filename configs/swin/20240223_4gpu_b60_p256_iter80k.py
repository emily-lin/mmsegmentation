backbone_norm_cfg = dict(requires_grad=True, type='LN')
checkpoint_file = 'pretrain/upernet_swinB_pretrain_ImageNet22K_ade20k_row4.pth'
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        124.95,
        124.95,
        124.95,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    std=[
        24.735,
        24.735,
        24.735,
    ],
    type='SegDataPreProcessor')
dataset_type = 'HeadCTDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=5000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(interval=1, type='HeadCTVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=512,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=128,
        init_cfg=dict(
            checkpoint=
            'pretrain/upernet_swinB_pretrain_ImageNet22K_ade20k_row4.pth',
            type='Pretrained'),
        mlp_ratio=4,
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_heads=[
            4,
            8,
            16,
            32,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        patch_size=4,
        pretrain_img_size=224,
        qk_scale=None,
        qkv_bias=True,
        strides=(
            4,
            2,
            2,
            2,
        ),
        type='SwinTransformer',
        use_abs_pos_embed=False,
        window_size=7),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            124.95,
            124.95,
            124.95,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
        std=[
            24.735,
            24.735,
            24.735,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            class_weight=[
                1.0,
                1.0,
                2.0,
                1.0,
                1.0,
                3.0,
                1.0,
            ],
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        sampler=dict(min_kept=20000, thresh=0.5, type='OHEMPixelSampler'),
        type='UPerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 7
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.00018, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            head=dict(lr_mult=5.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=80000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
ratio_range = (
    0.6,
    1.66,
)
reduce_zero_label = False
resume = False
test_cfg = dict(type='TestLoop')
test_crop_size = (
    512,
    512,
)
test_data_root = 'data/track'
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        data_root='data/track',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='HeadCTDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mDice',
        'mIoU',
    ], type='IoUROCMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
total_steps = 80000
train_cfg = dict(
    max_iters=80000, type='IterBasedTrainLoop', val_interval=16000)
train_crop_size = (
    256,
    256,
)
train_data_root = 'data/gotham'
train_dataloader = dict(
    batch_size=15,
    dataset=dict(
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        data_root='data/gotham',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.6,
                    1.66,
                ),
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.99, crop_size=(
                    256,
                    256,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='HeadCTDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_img_path = 'images/training'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.6,
            1.66,
        ),
        scale=(
            512,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.99, crop_size=(
        256,
        256,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
train_seg_map_path = 'annotations/training'
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        data_root='data/track',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='HeadCTDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mDice',
        'mIoU',
    ], type='IoUROCMetric')
val_img_path = 'images/validation'
val_interval = 16000
val_seg_map_path = 'annotations/validation'
vis_backends = [
    dict(type='LocalVisBackend'),
]
vis_interval = 1
visualizer = dict(
    name='visualizer',
    type='HeadCTVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
warmup_steps = 1500
work_dir = './work_dirs/20240223_4gpu_b60_p256_iter80k'
