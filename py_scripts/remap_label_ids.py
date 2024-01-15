import numpy as np
from PIL import Image
import os
import pdb

palette = np.asarray([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 0],
    [0, 0, 255]], dtype=np.uint8)

class_ratio = {
    2: 0.039713578521584234,
    3: 0.04793262230715194,
    4: 0.08633104268851109,
    5: 0.12378551411951855,
    6: 0.1971510761370291,
    7: 0.38564373016912146,
    8: 0.0023917739408469574,
    9: 0.07388203992002085,
    10: 0.043155424970272756,
    11: 9.9227262730126e-08,
    12: 1.3097998680376633e-05
    }

idtoname = {2: 'contusion', 3: 'epidural hematoma', 4: 'parafalcine subdural hematoma',
    5: 'intracerebral hematoma', 6: 'subarachnoid hemorrhage', 7: 'subdural hematoma (other than parafalcine)',
    8: 'traumatic axonal injury', 9: 'tentorial subdural hematoma', 10: 'intraventricular hemorrhage', 11: 'air',
    12: 'meningioma'}

def get_id_map():
  idmap = np.zeros(13)
  # Contusion/ICH
  one_idx = np.array([2, 5])
  idmap[one_idx] = 1
  # TAI
  idmap[8] = 2
  # Epidural
  idmap[3] = 3
  # Subdural
  four_idx = np.array([4, 7, 9])
  idmap[four_idx] = 4
  # SAH
  idmap[6] = 5
  # IVH
  idmap[10] = 6

  return idmap.astype(np.uint8)

def get_class_id_to_name_dict():
  class_id_to_name = {1: 'Contusion/ICH',
                      2: 'TAI',
                      3: 'Epidural',
                      4: 'Subdural',
                      5: 'SAH',
                      6: 'IVH'}
  return class_id_to_name

def makedirs(data_dir):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir) 
    os.makedirs(os.path.join(data_dir, 'train'))
    os.makedirs(os.path.join(data_dir, 'test'))

def remap_source_dir(source_dir, target_dir, idmap):
  source_gt = [gt for gt in os.listdir(source_dir) 
              if gt.endswith('.png')]
  for gt in source_gt:
    gt_full_path = os.path.join(source_dir, gt)
    gt_array = np.array(Image.open(gt_full_path))
    remapped_gt = idmap[gt_array]
    remapped_gt = Image.fromarray(remapped_gt)
    remapped_gt.save(os.path.join(target_dir, gt))
    print('Saved {} to {}.'.format(gt_full_path, 
                                   os.path.join(target_dir, gt)))

def main():
  mmseg_data = '/export/data/yuhlab1/emily/mmsegmentation/data'
  dataset = 'TRACK'
  gt_train_source = os.path.join(mmseg_data, dataset, 'gt_labels/train')
  gt_test_source = os.path.join(mmseg_data, dataset, 'gt_labels/test')
  gt_train_target = os.path.join(mmseg_data, dataset, 
                                 'gt_labels_remap/train')
  gt_test_target = os.path.join(mmseg_data, dataset, 'gt_labels_remap/test')

  makedirs(os.path.join(mmseg_data, dataset, 'gt_labels_remap'))
  idmap = get_id_map()

  remap_source_dir(gt_train_source, gt_train_target, idmap)
  remap_source_dir(gt_test_source, gt_test_target, idmap)

if __name__=="__main__":
  main()

