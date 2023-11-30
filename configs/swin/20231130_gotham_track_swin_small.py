_base_ = [
    './20231118_gotham_track_swin_tiny.py'
]

checkpoint_file = 'pretrain/upernet_swinS_pretrain_ImageNet1K_ade20k.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        depths=[2, 2, 18, 2]),
    decode_head=dict(in_channels=[96, 192, 384, 768]),
    auxiliary_head=dict(in_channels=384))
