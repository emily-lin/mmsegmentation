# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import sys
import os
import pdb
import tqdm
sys.path.insert(0, '/home/ubuntu/mmsegmentation')

from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot
import numpy as np
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out_dir', help='Path to output directory')
    parser.add_argument('--test_data', help='Test data directory')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()
    return args


def get_image_label_paths(data_dir, cutoff=None):
  test_dir = os.path.join(data_dir, 'images', 'test')
  test_images = [x for x in os.listdir(test_dir) if x.endswith('.png')]
  test_images = sorted(test_images)
  gt_labels = [x.replace('_Im', '_Gt') for x in test_images]
  test_image_paths = [os.path.join(data_dir, 'images', 'test', x) for x in test_images]
  gt_labels_paths = [os.path.join(data_dir, 'gt_labels', 'test', x) for x in gt_labels]
  if cutoff:
    test_image_paths = test_image_paths[:cutoff]
    gt_labels_paths = gt_labels_paths[:cutoff]

  return test_image_paths, gt_labels_paths


def main():
  args = parse_args()

  # build the model from a config file and a checkpoint file
  model = init_model(args.config, args.checkpoint, device=args.device)
  if args.device == 'cpu':
      model = revert_sync_batchnorm(model)

  test_image_paths, gt_labels_paths = get_image_label_paths(args.test_data, cutoff=None)
  for im_path, gt_path in tqdm.tqdm(zip(test_image_paths, gt_labels_paths)):
    np_img = np.array(Image.open(im_path))
    original_image = np.tile(np_img[:, :, 1:2], (1, 1, 3))
    result = inference_model(model, np_img)
    # Store GT mask.
    gt_mask = np.array(Image.open(gt_path))
    gt_mask = torch.from_numpy(gt_mask)
    result.gt_sem_seg = PixelData(data=gt_mask)
    # show the results
    if not os.path.exists(args.out_dir):
      os.makedirs(args.out_dir)
    out_file = os.path.join(args.out_dir, im_path.split('/')[-1].replace('.png', '.jpg'))
    show_result_pyplot(
        model,
        original_image,
        result,
        opacity=args.opacity,
        draw_gt=True,
        show=False,
        out_file=out_file)


if __name__ == '__main__':
    main()
