import os


def main():
    txt_file = 'debug/overfit_data/20231120_sampled_gt.txt'
    output_dir = 'overfit_gotham'
    source_im_dir = 'data/gotham/images/training'
    source_ann_dir = 'data/gotham/annotations/training'

    os.makedirs(f'debug/overfit_data/{output_dir}/images/training', exist_ok=True)
    os.makedirs(f'debug/overfit_data/{output_dir}/annotations/training', exist_ok=True)
    
    with open(txt_file, 'r') as f:
        tfiles = f.readlines()
        tfiles = [t.strip('\n') for t in tfiles]

        for fname in tfiles:
            im_fname = fname.replace('Gt', 'Im') 
            os.system(f'cp {source_im_dir}/{im_fname} debug/overfit_data/{output_dir}/images/training/')
            os.system(f'cp {source_ann_dir}/{fname} debug/overfit_data/{output_dir}/annotations/training/')

if __name__ == '__main__':
    main()
