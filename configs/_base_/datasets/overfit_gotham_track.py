# dataset settings
dataset_type = 'HeadCTDataset'
train_data_root = 'data/overfit_gotham'
test_data_root = 'data/overfit_gotham'
reduce_zero_label = False # Must set to False (True will break evaluation on existing checkpoint).
ratio_range = (1.0, 1.0)
test_crop_size = (512, 512)
train_seg_map_path = 'annotations/training'
val_seg_map_path = 'annotations/training'
train_img_path = 'images/training'
val_img_path = 'images/training'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
    dict(
        type='RandomResize',
        scale=test_crop_size,
        ratio_range=ratio_range,
        keep_ratio=True),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=test_crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=train_data_root,
        data_prefix=dict(
            img_path=train_img_path, seg_map_path=train_seg_map_path),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=test_data_root,
        data_prefix=dict(
            img_path=val_img_path,
            seg_map_path=val_seg_map_path),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = val_evaluator
