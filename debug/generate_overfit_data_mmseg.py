import os


def main():
    mmsegmentation = '/home/ubuntu/mmsegmentation'
    txt_file = os.path.join(mmsegmentation, 'debug/20231120_pos_gotham.txt')
    output_dir = 'gotham_pos'
    source_im_dir = os.path.join(mmsegmentation, 'data/gotham/images/training')
    source_ann_dir = os.path.join(mmsegmentation, 'data/gotham/annotations/training')

    os.makedirs(f'/home/ubuntu/mmsegmentation/data/{output_dir}/images/training', exist_ok=True)
    os.makedirs(f'/home/ubuntu/mmsegmentation/data/{output_dir}/annotations/training', exist_ok=True)
    
    with open(txt_file, 'r') as f:
        tfiles = f.readlines()
        tfiles = [t.strip('\n') for t in tfiles]

        for fname in tfiles:
            im_fname = fname.replace('Gt', 'Im') 
            os.system(f'cp {source_im_dir}/{im_fname} /home/ubuntu/mmsegmentation/data/{output_dir}/images/training/')
            os.system(f'cp {source_ann_dir}/{fname} /home/ubuntu/mmsegmentation/data/{output_dir}/annotations/training/')

if __name__ == '__main__':
    main()
