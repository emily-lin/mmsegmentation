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
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512))
