import os
import numpy as np
from PIL import Image
import pdb

data_root = '/home/ubuntu/mmsegmentation/data'
gotham = os.path.join(data_root, 'gotham/annotations/training')
track = os.path.join(data_root, 'track/annotations/validation')


def get_class_ids(data):
    """
    Iterate through all gt labels to get all class ids. Initially written
    to debug whether gotham class labels are 0-6 or 0-12.

    Input: 
        data (str): directory name containing ground truth files

    Output:
        class_ids (list): list of all unique class labels.

    """
    all_gt = os.listdir(data)
    class_ids = set()

    for gt in all_gt:
        np_gt = np.array(Image.open(os.path.join(data, gt)))
        gt_values = np.unique(np_gt).tolist()
        print('Processing {}... Unique gt values: {}'.format(gt, gt_values))
        class_ids.update(gt_values)

    return class_ids

if __name__ == '__main__':
    gotham_ids = get_class_ids(gotham)
    track_ids = get_class_ids(track)
    print('Gotham ids: {}'.format(gotham_ids))
    print('TRACK ids: {}'.format(track_ids))

