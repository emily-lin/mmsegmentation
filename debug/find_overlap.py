import os
import csv
import pandas as pd
import pdb

mmsegmentation = '/home/ubuntu/mmsegmentation'
isolated_sah = os.path.join(mmsegmentation, 'debug/SAH_isolated_TRACK_final.csv')
track = os.path.join(mmsegmentation, 'data/track/images/validation')
frostbite = os.path.join(mmsegmentation, 'debug/frostbite_0802_images.txt')

# Read isolated_sah, track, and frostbite into lists
with open(isolated_sah, 'r') as file:
    csv_reader = csv.reader(file)
    sah_list = []
    for row in csv_reader:
        sah_list.append(row[0])

track_list = os.listdir(track)

with open(frostbite, 'r') as file:
    frostbite_list = [line.strip('\n') for line in file.readlines()]


# Identify exams on SAH list that I have segmentations for in TRACK
sah_with_seg = set()
for image in track_list:
    for idx in sah_list:
        if image.startswith(idx) or image.startswith(idx.replace('-', '_')):
            sah_with_seg.add(idx)
sah_with_seg = list(sah_with_seg)

# Identify overlap
overlap = set()
for image in frostbite_list:
    for idx in sah_with_seg:
        if image.startswith(idx) or image.startswith(idx.replace('-', '_')):
            overlap.add(idx)

print('Exams that are overlapping and should be excluded: {}'.format(overlap))
pdb.set_trace()
