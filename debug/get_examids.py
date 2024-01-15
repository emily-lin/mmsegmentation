import os
import pdb

def get_examids_txt(txt_file):
    examids = set()

    with open(txt_file, 'r') as file:
        images = [line.strip('\n') for line in file.readlines()]
    for image in images:
        if image.startswith('PTHTRNE'):
            examid = image.split('_')[0]
            examids.add(examid)
        else:
            examid = image[:7]
            examids.add(examid)
    examids = sorted(list(examids))

    return examids

def get_examids_dir(directory):
    examids = set()
    images = os.listdir(directory)
    
    for image in images:
        if image.startswith('PTHTRNE'):
            examid = image.split('_')[0]
            examids.add(examid)
        else:
            examid = image[:7]
            examids.add(examid)
    examids = sorted(list(examids))
    
    return examids

def find_overlap(exam_list1, exam_list2):
    overlap = set()
    for exam1 in exam_list1:
        for exam2 in exam_list2:
            if exam1 == exam2:
                overlap.add(exam1)
    overlap = sorted(list(overlap))

    return overlap

def write_text_file(examid_list, output):
    with open(output, 'w') as file:
        for examid in examid_list:
            file.write(examid + '\n')

if __name__ == "__main__":
    mmsegmentation = '/home/ubuntu/mmsegmentation'
    frostbite = os.path.join(mmsegmentation, 'debug/frostbite_0802_images.txt')
    gotham = os.path.join(mmsegmentation, 'data/gotham/images/training')
    track = os.path.join(mmsegmentation, 'data/track/images/validation')

    frostbite_output = os.path.join(mmsegmentation, 'debug/frostbite_examids.txt')
    gotham_output = os.path.join(mmsegmentation, 'debug/gotham_examids.txt')
    track_output = os.path.join(mmsegmentation, 'debug/track_examids.txt')
    gotham_track_overlap_output = os.path.join(mmsegmentation, 'debug/gotham_track_overlap.txt')
    frostbite_track_overlap_output = os.path.join(mmsegmentation, 'debug/frostbite_track_overlap.txt')
    

    frostbite_examids = get_examids_txt(frostbite)
    gotham_examids = get_examids_dir(gotham)
    track_examids = get_examids_dir(track)

    gotham_track_overlap = find_overlap(gotham_examids, track_examids)
    frostbite_track_overlap = find_overlap(frostbite_examids, track_examids)

    write_text_file(frostbite_examids, frostbite_output)
    write_text_file(gotham_examids, gotham_output)
    write_text_file(track_examids, track_output)
    write_text_file(frostbite_track_overlap, frostbite_track_overlap_output)

