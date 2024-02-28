from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import get_palette
import progressbar
import numpy as np
from PIL import Image, ImageDraw
import mmcv
import os
import pdb

from mmengine.structures import PixelData


mmseg = '/home/ubuntu/mmsegmentation/'
temp = os.path.join(mmseg, 'temp')
test_dir = os.path.join(mmseg, 'data/track/images/validation')
gt_dir = os.path.join(mmseg, 'data/track/annotations/validation')
config_file = os.path.join(mmseg, 'configs/swin/20240223_4gpu_b60_p256_iter80k.py')
checkpoint_file = os.path.join(mmseg, 'work_dirs/20240223_4gpu_b60_p256_iter80k/iter_80000.pth')
pred_palette = [[0, 0, 0], [0, 255, 0], [255, 0, 0,], [0, 0, 255], 
                [128, 128, 0], [128, 0, 128], [0, 128, 128]]
gt_palette = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255],
              [128, 128, 0], [128, 0, 128], [0, 128, 128]]  # BGR for OpenCV.
test_imgs = sorted([x for x in os.listdir(test_dir) if x.endswith('.png')])
out_dir = os.path.join(mmseg, 'visualizations/20240223_4gpu_b60_p256_iter80k')
cls_names = ['Background', 'Contusion', 'Petechial', 'Epidural', 'Subdural',
             'Subarachnoid', 'Intraventricular']

for cls_name, palette in zip(cls_names, pred_palette):
  palette_im = np.ones((128, 128, 1), np.uint8) * np.array(palette, np.uint8)[None, None, :]  # (128, 128, 3).
  palette_im = Image.fromarray(palette_im)
  out_path = os.path.join(temp, f'{cls_name}_color.png')
  palette_im.save(out_path)

for img in progressbar.progressbar(test_imgs):
  img_path = os.path.join(test_dir, img)
  np_img = np.array(Image.open(img_path))
  original_image = np.tile(np_img[:, :, 1:2], (1, 1, 3))
  # print(img, original_image[0, 0])
  # original_image = Image.fromarray(original_image)
  # original_image_path = os.path.join(temp, img)
  # original_image.save(original_image_path)

  gt = img.replace('_Im', '_Gt')
  gt_path = os.path.join(gt_dir, gt)
  gt_labels = np.array(Image.open(gt_path))
  if not np.any(gt_labels):
  # if np.any(gt_labels):
    continue

  model = init_model(config_file, checkpoint_file, device = 'cuda:0')
  prediction = inference_model(model, img_path)
  gt_seg = PixelData(data=gt_labels)
  prediction.gt_sem_seg = gt_seg
  # if not np.any(prediction.pred_sem_seg.data.cpu().numpy()):
    # continue
  # GT is on the left, Prediction on the right.
  # https://github.com/open-mmlab/mmsegmentation/blob/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8/mmseg/visualization/local_visualizer.py#L271
  pred_overlaid_im = show_result_pyplot(model, original_image, prediction,
                                        opacity = 0.5, draw_gt = True, show = False, withLabels = False,
                                        out_file = os.path.join(out_dir, img))
