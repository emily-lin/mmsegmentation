import numpy as np
from PIL import Image
import os
import pdb

def count_pos_pixels(directory: str) -> int:
  '''
  Count the total number of positive pixels across the whole training split. 
  Args:
    directory (str): absolute file path to training directory

  Returns:
    pos_pixels (int): total positive pixel count across training set. 
  '''
  pos_pixels = 0
  os.chdir(directory)
  gt_labels = [file for file in os.listdir(directory) if file.endswith('.png')]
  for i, gt in enumerate(gt_labels):
    if i % 1000 == 0:
      print(f'Loading gt image {i}: {gt}. Count: {pos_pixels}')
    gt_image = Image.open(gt)
    gt_array = np.array(gt_image)
    pos_pixels += np.sum(gt_array > 0)

  print('Final pixel count:', pos_pixels)
  return pos_pixels

def pos_pixels_per_class(pos_pixels: int, pix_distributions: list) -> list:
  '''
  Calculate the number of positive pixels per class
  Args:
    pos_pixels (int): total positive pixel count across training set
    pix_distributions (list): list of (class_name, distribution) across each of the classes
  Return:
    pos_pix_per_class (list): list of (class name, total class positive pixels)
  '''
  pos_pix_per_class = []
  for (class_name, distribution) in pix_distributions:
    class_pos = pos_pixels * distribution
    pos_pix_per_class.append((class_name, class_pos))

  return pos_pix_per_class

def image_instances_per_class(directory: str, n_classes: int):
  instances_per_class = np.zeros(n_classes)

  os.chdir(directory)
  gt_labels = [file for file in os.listdir(directory) if file.endswith('.png')]
  for idx, gt_label in enumerate(gt_labels):
    if idx % 1000 == 0:
      print('Loading gt image {}: {}'.format(idx, gt_label))
    gt_image = Image.open(gt_label)
    gt_array = np.array(gt_image)
    instances = np.unique(gt_array)
    for instance in instances:
      if instance == 0:
        pass
      else:
        instances_per_class[instance - 1] += 1

  pdb.set_trace()
  return instances_per_class

def class_reweights(instances_per_class: list, beta: list, n_classes: int):
  '''
  Compute the class weights for each value of beta
  Args:
    instances_per_class (list): list of instances per class
    beta (list): list of all beta values to try
    n_classes (int): number of positive classes (excluding negative class)
  '''
  bg_value = 0.3
  lower_bound = 0.01
  upper_bound = 10.0
  for value in beta:
    w = np.zeros(n_classes + 1)
    for idx in range(1, n_classes + 1):
      w[idx] = (1.0 - value) / (1.0 - np.power(value, instances_per_class[idx - 1]))
      # w[idx] = (1.0 - value) / (1.0 - np.power(value, pos_pix_per_class[idx - 1][1]))
      # w[idx] = (1.0 - value) / (1.0 - np.exp(np.log(value) * pos_pix_per_class[idx - 1][1]))
    
    wnorm = np.mean(w[1:])
    # print('Before normalization', w)
    # print(f'W normalization {wnorm}')
    w = w / wnorm
    # print('After normalization', w)
    w = np.clip(w, lower_bound, upper_bound)
    w[0] = bg_value
    print('Beta value: {}, clipped weights: {}'.format(value, w))

def main():
  directory = '/export/data/yuhlab1/emily/mmsegmentation/data/gotham_cv0/gt_labels/train'
  pix_distributions = [('Contusion/ICH', 0.1635), ('Petechial', 0.0024), 
                       ('Epidural', 0.0479), ('Subdural', 0.5458), 
                       ('Subarachnoid', 0.1971), ('Intraventricular', 0.0432)]
  instances_per_class = [2201., 394., 736., 8899., 7004., 963.]
  beta = [0.9, 0.99, 0.999, 0.9999, 0.99999]
  n_classes = 6

  # pos_pixels = 18123714
  # pos_pixels = count_pos_pixels(directory)
  # pos_pix_per_class = pos_pixels_per_class(pos_pixels, pix_distributions)
  # class_reweights(pos_pix_per_class, beta, n_classes)

  # instances_per_class = image_instances_per_class(directory, n_classes)
  # instances_per_class = instances_per_class.toList()
  class_reweights(instances_per_class, beta, n_classes)

if __name__=="__main__":
  main()
