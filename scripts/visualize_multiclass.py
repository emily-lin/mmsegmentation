from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import numpy as np
from PIL import Image, ImageDraw
import mmcv
import os
import pdb

mmseg = '/export/data/yuhlab1/emily/mmsegmentation/'
out_dir = '/export/data/yuhlab1/emily/mmsegmentation/vis/20220202_vis_gothamtest'
temp = '/export/data/yuhlab1/emily/temp'
split = 'test'
test_dir = os.path.join(mmseg, 'data/TRACK/images/{}'.format(split))
gt_dir = os.path.join(mmseg, 'data/TRACK/gt_labels/{}'.format(split))
config_file = os.path.join(mmseg, 'configs/swin/20220119_mmseg_mcls6.py')
checkpoint_file = os.path.join(mmseg, 'work_dirs/20220119_mmseg_mcls6/latest.pth')
pred_palette = [[0, 0, 0], [0, 255, 0], [255, 0, 0,], [0, 0, 255], 
                [128, 128, 0], [128, 0, 128], [0, 128, 128]]
gt_palette = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255],
              [128, 128, 0], [128, 0, 128], [0, 128, 128]]  # BGR for OpenCV.
test_imgs = [x for x in os.listdir(test_dir) if x.endswith('.png')]
test_imgs = sorted(test_imgs)
cls_names = ['Background', 'Contusion', 'Petechial', 'Epidural', 'Subdural',
             'Subarachnoid', 'Intraventricular']

for cls_name, palette in zip(cls_names, pred_palette):
  palette_im = np.ones((128, 128, 1), np.uint8) * np.array(palette, np.uint8)[None, None, :]  # (128, 128, 3).
  palette_im = Image.fromarray(palette_im)
  out_path = os.path.join(out_dir, f'{cls_name}_color.png')
  palette_im.save(out_path)

for img in test_imgs:
  img_path = os.path.join(test_dir, img)
  np_img = np.array(Image.open(img_path))
  original_image = np_img[:, :, 1]
  # print(img, original_image[0, 0])
  original_image = Image.fromarray(original_image)
  original_image_path = os.path.join(temp, img)
  original_image.save(original_image_path)

  gt = img.replace('_Im', '_Gt')
  gt_path = os.path.join(gt_dir, gt)
  gt_labels = np.array(Image.open(gt_path))
  if not np.any(gt_labels):
    continue

  model = init_segmentor(config_file, checkpoint_file, device = 'cuda:0')
  prediction = inference_segmentor(model, img_path)
  pred_overlaid_im = model.show_result(original_image_path, prediction,
                                       palette = pred_palette, opacity = 0.5)
  gt_overlaid_im = model.show_result(original_image_path, [gt_labels],
                                     palette = gt_palette, opacity = 0.5)
  join_im = np.concatenate((gt_overlaid_im, pred_overlaid_im), axis=1)
  join_im = Image.fromarray(join_im)
  out_path = os.path.join(out_dir, img)
  join_im.save(out_path)
  print('Save {} to {}.'.format(img, out_path))
  print('\n')