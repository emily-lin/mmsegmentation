import os
from PIL import Image
import numpy as np
import random
import pdb

data = '/home/ubuntu/mmsegmentation/data/'
gotham = os.path.join(data, 'gotham/annotations/training')

def get_positive_gt(data, num):
    """
    Get all positive gt and a desired subsample from the dataset by 
    reading gt files.

    Input:
        data (str): name of directory containing all gt files. 
        num (int): number of images you want to sample.
    
    Output:     
        positive_gt (list): list of all positive files
        sampled_gt (list): list of all sampled files

    """
    all_gt = os.listdir(data)
    positive_gt = []

    for gt in all_gt:
        print('Processing file {}...'.format(gt))
        np_gt = np.array(Image.open(os.path.join(gotham, gt)))
        if len(np.unique(np_gt).tolist()) > 1:
            positive_gt.append(gt)

    print('Sampling gt_files...')
    sampled_gt = random.sample(positive_gt, num)
    
    return positive_gt, sampled_gt


def write_text_files(sampled_gt, output):
    with open(output, 'w') as file:
        for gt in sampled_gt:
            file.write(gt + '\n')
    

if __name__ == '__main__':
    output = 'overfit_data/20231120_sampled_gt.txt'

    positive_gt, sampled_gt = get_positive_gt(data = gotham, num = 25)
    print('Positive gt: {}'.format(positive_gt))
    print('Number of positive gt files: {}'.format(len(positive_gt)))
    print('Sampled gt: {}'.format(sampled_gt))
    print('Number of sampled gt files: {}'.format(len(sampled_gt)))
    
    write_text_files(sampled_gt = sampled_gt, output = output)

