import os
import csv
import json
import pandas as pd
import numpy as np
import progressbar
from typing import Optional
from PIL import Image
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import pdb

def get_examid(gt_name: str) -> str:
  separators = gt_name.split('_')
  for separator in separators:
    if 'Obj' in separator:
      examid = separator

  return examid

def get_image_gt_volumes(gt_name, pos_classes: int):
  """Compute gt or prediction multiclass volume.

  Args:
    gt_name: A path to load the image or a numpy array.

  Returns:
    mcls_sum: The multiclass volume of this image.
  """
  mcls_sum = np.zeros(pos_classes)

  if isinstance(gt_name, str):
    gt_image = Image.open(gt_name)
    gt_array = np.array(gt_image)
  elif isinstance(gt_name, np.ndarray):
    gt_array = gt_name
  else:
    raise ValueError('Unknown gt_name type:', type(gt_name))

  if len(gt_array.shape) == 2:  # Integer index array from GT or Pred.
    # gt_array shape: [height, width].
    for class_idx in range(1, 1 + pos_classes):
      class_sum = np.sum(gt_array == class_idx)
      mcls_sum[class_idx - 1] = class_sum
  
  else:
    raise ValueError('Invalid shape for gt_array:', gt_array.shape)

  return mcls_sum 

def compute_volume_error(gt_volumes: dict, prediction_volumes: dict):
  exam_error_dict = {}

  examids = gt_volumes.keys()
  for examid in examids:
    gt_volume = gt_volumes[examid]
    prediction_volume = prediction_volumes[examid]
    if gt_volume > 0:
        error = np.absolute((prediction_volume / gt_volume) - 1)
        exam_error_dict[examid] = error

  return exam_error_dict

def aggregate_exam_error(exam_error_dict: dict, pos_classes: int) -> np.array:
  """Aggregate the exam errors."""
  mcls_sum = np.zeros(pos_classes)
  mcls_num_exams = np.zeros(pos_classes)
  for exam_id, error in exam_error_dict.items():
    mcls_sum += (error != -1) * error  # Set the invalid volume to 0.
    mcls_num_exams += (error != -1)

  return mcls_sum / mcls_num_exams

def main():
  mmsegmentation = '/home/ubuntu/mmsegmentation'
  mcls_names = ['Contusion/ICH', 'Petechial', 'EDH', 'SDH', 'SAH', 'IVH']
  # Pixel ratio for multiclass hemorrhages. First element is dummy padding for background.
  # mcls_pix_ratio = np.array([0., 0.1635, 0.0024, 0.0479, 0.5458, 0.1971, 0.0432])
  # mcls_threshold = np.arange(0.1, 1.1, 0.2)
  # mcls_threshold = np.zeros(1)
  test_gt_dir = os.path.join(mmsegmentation, 'data/track/annotations/validation')
  test_im_dir = os.path.join(mmsegmentation, 'data/track/images/validation')
  checkpoint = os.path.join(mmsegmentation, 'work_dirs/20240108_mixed_loss/iter_40000.pth')
  config = os.path.join(mmsegmentation, 'configs/swin/20231130_gotham_track_swin_base.py')
  temp = os.path.join(mmsegmentation, 'temp')
  dir_to_save = os.path.join(mmsegmentation,
                             'work_dirs/20240108_mixed_loss/inferences')
  pred_output = os.path.join(mmsegmentation, 'temp/202410111_track_pred_volumes_bestmodel.json')
  gt_output = os.path.join(mmsegmentation, 'temp/20240111_track_gt_volumes_bestmodel.json')
  error_output = os.path.join(mmsegmentation, 'temp/20240111_track_error_bestmodel.json')
  pos_classes = 6
  gt_mcls_volumes = {}
  prediction_mcls_volumes = {}

  all_gt = sorted([file for file in sorted(os.listdir(test_gt_dir)) if '_Gt' in file])
  all_images = sorted([file for file in sorted(os.listdir(test_im_dir)) if '_Im' in file])

  # Get binary ground truth volumes
  print('Getting binary ground truth volumes...')
  for gt in progressbar.progressbar(all_gt):
    examid = get_examid(gt)
    mcls_sum = get_image_gt_volumes(os.path.join(test_gt_dir, gt), pos_classes)
    if examid not in gt_mcls_volumes.keys():
      gt_mcls_volumes[examid] = mcls_sum
    else:
      gt_mcls_volumes[examid] += mcls_sum

  gt_binary_volumes = {key: int(sum(values)) for key, values in gt_mcls_volumes.items()}
    
  # Get binary prediction volumes
  model = init_model(config, checkpoint, device = 'cuda:0')
  print('Getting binary prediction volumes...')
  for image in progressbar.progressbar(all_images):
    examid = get_examid(image)
    prediction_path = image.replace('_Im', '_Pred').replace('.png', '.npy')
    prediction_path = os.path.join(dir_to_save, prediction_path)
    if os.path.exists(prediction_path):
        prediction = np.load(open(prediction_path, 'rb'))
    else:
        prediction = inference_model(model, os.path.join(test_im_dir, image))
        prediction = np.array(prediction.pred_sem_seg.data.cpu())[0]
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save)
        # with open(prediction_path, 'wb') as fid:
          # np.save(fid, prediction)
          # print(f'Saving image {prediction_path}')
    mcls_sum = get_image_gt_volumes(prediction, pos_classes)
    
    if examid not in prediction_mcls_volumes.keys():
        prediction_mcls_volumes[examid] = mcls_sum
    else:
        prediction_mcls_volumes[examid] += mcls_sum

  prediction_binary_volumes = {key: int(sum(values)) for key, values in prediction_mcls_volumes.items()}

  exam_error_dict = compute_volume_error(gt_binary_volumes, prediction_binary_volumes)
  
  # Find average exam error
  avg_error = np.mean([error for examid, error in exam_error_dict.items()])
  print('Average error: {}'.format(avg_error))
  # avg_error = aggregate_exam_error(exam_error_dict, pos_classes)
  pdb.set_trace()  
  
  # Write binary predictions to csv
  # json.dump(prediction_binary_volumes, open(pred_output, 'w'), separators = ('\n', ':'))  
  # json.dump(gt_binary_volumes, open(gt_output, 'w'), separators = ('\n', ':'))  
  # json.dump(exam_error_dict, open(error_output, 'w'), separators = ('\n', ':'))  

if __name__=="__main__":
  main()
