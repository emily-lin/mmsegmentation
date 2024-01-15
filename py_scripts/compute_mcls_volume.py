import os
import numpy as np
import progressbar
from typing import Optional
from PIL import Image
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import mmcv
import pdb

def get_examid(gt_name: str) -> str:
  separators = gt_name.split('_')
  for separator in separators:
    if 'Obj' in separator:
      examid = separator

  return examid

def get_image_gt_volumes(gt_name: str, pos_classes: int,
                         mcls_thr: Optional[np.ndarray] = None):
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
  elif len(gt_array.shape) == 3:  # Float array of class probabilities from Pred.
    # gt_array shape: [num_classes, height, width].
    assert pos_classes == gt_array.shape[0] - 1
    # Threshold the multiclass pixel confidence scores.
    assert isinstance(mcls_thr, np.ndarray) or isinstance(mcls_thr, np.float64)
    if len(mcls_thr.shape) == 1:
      gt_array = gt_array * (gt_array > mcls_thr[:, None, None])
    elif len(mcls_thr.shape) == 0:
      gt_array = (gt_array > mcls_thr).astype(np.float64)
    else:
      raise ValueError('Invalid threshold shape:', mcls_thr.shape)
    # Exclude the first background class.
    mcls_sum = np.sum(gt_array[1:], axis=(1, 2))
  else:
    raise ValueError('Invalid shape for gt_array:', gt_array.shape)

  return mcls_sum

def compute_volume_error(gt_volumes: dict, prediction_volumes: dict, pos_classes: int):
  volume_error = np.zeros(pos_classes)
  exam_error_dict = {}

  examids = gt_volumes.keys()
  for examid in examids:
    gt_volume = gt_volumes[examid]
    prediction_volume = prediction_volumes[examid]
    error = np.absolute((prediction_volume / gt_volume) - 1)
    # Mask out the absent classes to -1.
    error = np.where(gt_volume > 0, error, -1)
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
  mmsegmentation = '/export/data/yuhlab1/emily/mmsegmentation'
  mcls_names = ['Contusion/ICH', 'Petechial', 'EDH', 'SDH', 'SAH', 'IVH']
  # Pixel ratio for multiclass hemorrhages. First element is dummy padding for background.
  # mcls_pix_ratio = np.array([0., 0.1635, 0.0024, 0.0479, 0.5458, 0.1971, 0.0432])
  # mcls_threshold = np.arange(0.1, 1.1, 0.2)
  # mcls_threshold = np.zeros(1)
  test_gt_dir = os.path.join(mmsegmentation, 'data/TRACK/gt_labels/test')
  test_im_dir = os.path.join(mmsegmentation, 'data/TRACK/images/test')
  checkpoint = os.path.join(mmsegmentation, 'work_dirs/20220119_mmseg_mcls6/latest.pth')
  config = os.path.join(mmsegmentation, 'configs/swin/20220121_test_track.py')
  temp = '/export/data/yuhlab1/emily/temp'
  dir_to_save = os.path.join(mmsegmentation,
                             'work_dirs/20220119_mmseg_mcls6/inferences')
  pos_classes = 6
  num_images = 10
  gt_volumes = {}
  prediction_volumes = {}

  all_gt = [file for file in sorted(os.listdir(test_gt_dir)) if '_Gt' in file]
  all_images = [file for file in sorted(os.listdir(test_im_dir)) if '_Im' in file]

  # Get multiclass ground truth volumes
  print('Getting multiclass ground truth volumes...')
  for gt in progressbar.progressbar(all_gt):
    examid = get_examid(gt)
    mcls_sum = get_image_gt_volumes(os.path.join(test_gt_dir, gt), pos_classes)
    if examid not in gt_volumes.keys():
      gt_volumes[examid] = mcls_sum
    else:
      gt_volumes[examid] += mcls_sum

  # Get multiclass prediction volumes
  thr_avg_error = []
  for mcls_thr in mcls_threshold:
    model = init_segmentor(config, checkpoint, device = 'cuda:0')
    print(f'Getting multiclass prediction volumes with threshold {mcls_thr}')
    for image in progressbar.progressbar(all_images):
      examid = get_examid(image)
      prediction_path = image.replace('_Im', '_Pred').replace('.png', '.npy')
      prediction_path = os.path.join(dir_to_save, prediction_path)
      if os.path.exists(prediction_path):
        prediction = np.load(open(prediction_path, 'rb'))
      else:
        prediction = inference_segmentor(model, image)[0]
        with open(prediction_path, 'wb') as fid:
          np.save(fid, prediction)
          print(f'Saving image {prediction_path}')

      mcls_sum = get_image_gt_volumes(prediction, pos_classes, mcls_thr)
      if examid not in prediction_volumes.keys():
        prediction_volumes[examid] = mcls_sum
      else:
        prediction_volumes[examid] += mcls_sum

    exam_error_dict = compute_volume_error(gt_volumes, prediction_volumes, pos_classes)
    avg_error = aggregate_exam_error(exam_error_dict, pos_classes)
    thr_avg_error.append(avg_error[None, :])
    print('=============================')
    print(f'Multiplier {mcls_thr}, Average Volume Error:')
    print('=============================')
    for bleed_type, error in zip(mcls_names, avg_error.tolist()):
      print(f'{bleed_type}: {error}')

  thr_avg_error = np.concatenate(thr_avg_error, axis=0)
  best_thr_idx = np.argmin(thr_avg_error, axis=0)
  best_mcls_thr = mcls_threshold[best_thr_idx]
  best_mcls_error = np.amin(thr_avg_error, axis=0)
  print('=============================')
  print(f'Best Average Volume Error:')
  print('=============================')
  for bleed_type, best_thr, error in zip(
      mcls_names, best_mcls_thr, best_mcls_error.tolist()):
    print(f'{bleed_type} - Threshold: {best_thr}, Error: {error}')

if __name__=="__main__":
  main()
