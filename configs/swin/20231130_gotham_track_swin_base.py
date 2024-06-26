_base_ = [
    './20231118_gotham_track_swin_tiny.py'
]
checkpoint_file = 'pretrain/upernet_swinB_pretrain_ImageNet22K_ade20k_row4.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[128, 256, 512, 1024],
        sampler = dict(type = 'OHEMPixelSampler', thresh = 0.5, min_kept = 20000),
        loss_decode = [
            dict(type = 'CrossEntropyLoss', loss_name = 'loss_ce', loss_weight = 3.0), 
            dict(type = 'DiceLoss', loss_name = 'loss_dice', loss_weight = 1.0)]),
    auxiliary_head=dict(in_channels=512,
        loss_decode = [
            dict(type = 'CrossEntropyLoss', loss_name = 'loss_ce', loss_weight = 3.0),
            dict(type = 'DiceLoss', loss_name = 'loss_dice', loss_weight = 1.0)]),
        )

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
        type='OptimWrapper',
        optimizer = dict(
            type = 'AdamW', lr = 1.8E-4, betas = (0.9, 0.999), weight_decay = 0.01),
        paramwise_cfg = dict(
            custom_keys = {
                'absolute_pos_embed': dict(decay_mult = 0.),
                'relative_position_bias_table': dict(decay_mult = 0.),
                'norm': dict(decay_mult = 0.),
                'head': dict(lr_mult = 5.)
            }))

train_dataloader = dict(batch_size = 60)
