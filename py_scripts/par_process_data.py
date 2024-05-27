import os
import csv
import os.path as osp
from os.path import exists, join, split
import numpy as np
import pickle
import pdb
from PIL import Image
import json
from multiprocessing import Pool
from functools import partial
import scipy.sparse
#from skimage.measure import label,perimeter
#from skimage.filters.rank import median
#from skimage.morphology import disk

SKIP_ZERO_THR = 0.95
PIX_LOW = -40.
PIX_HIGH = 90.
MAX_INT = 255.
bg_value = int(MAX_INT*(-PIX_LOW)/(PIX_HIGH-PIX_LOW))
dataname = 'bermuda'
output_name = dataname
num_proc = 8
use_gt = False


def quick_convert_csv2png(f):
  """
  Check how many CSV are converted to png.
  """
  data_dir = '/data/20240524_unlabeled_grant_data/NEW_CSV_FILES_EMILY_FOLLOWUP_CTs'
  fpath = os.path.join(data_dir,f)
  output_dir = '/data/20240524_unlabeled_grant_data/images'
  center_frame = np.loadtxt(fpath,delimiter=',',dtype=np.float32) 
  #Check the amount of zeros in the image
  if np.sum(center_frame==0)>SKIP_ZERO_THR * center_frame.size:
    print("Excessive zeros {}".format(fpath))
    return (None,None,None)

  else:
    return (1, 1, 1)

def convert_csv2png(f, data_dir):
  """
  Read in csv data and save as png data
  """
  fpath = os.path.join(data_dir,f)
  output_dir = '/data/20240524_unlabeled_grant_data/images'
  print(fpath)
  
  #Check the fake ground truth frame
  if 'fake.csv' in f:
    # print("Fake ground truth")
    return (None,None,None)
  center_frame = np.loadtxt(fpath,delimiter=',',dtype=np.float32) 
  
  if use_gt:
    gtpath = fpath.replace("_Im_","_Gt_")
    gt_frame = np.loadtxt(gtpath,delimiter=',',dtype=np.uint8) 

  #Check the amount of zeros in the image
  if np.sum(center_frame==0)>SKIP_ZERO_THR * center_frame.size:
    # print("Excessive zeros {}".format(fpath))
    return (None,None,None)

  # print('Running {}'.format(fpath))
  fpath_id = int(fpath[-7:-4])
  up_fpath = fpath.split('_Im_')[0]+'_Im_{0:03d}.csv'.format(fpath_id-1)
  if os.path.exists(up_fpath):
    up_frame = np.loadtxt(up_fpath,delimiter=',',dtype=np.float32)
  else:
    up_fpath = fpath.split('_Im_')[0]+'_Im_{0:03d}fake.csv'.format(fpath_id-1)
    if os.path.exists(up_fpath):
      up_frame = np.loadtxt(up_fpath,delimiter=',',dtype=np.float32)
    else:
      up_frame = np.zeros(center_frame.shape)

  down_fpath = fpath.split('_Im_')[0]+'_Im_{0:03d}.csv'.format(fpath_id+1)
  if os.path.exists(down_fpath):
    down_frame = np.loadtxt(down_fpath,delimiter=',',dtype=np.float32)
  else:
    down_fpath = fpath.split('_Im_')[0]+'_Im_{0:03d}.csv'.format(fpath_id+1)
    if os.path.exists(down_fpath):
      down_frame = np.loadtxt(down_fpath,delimiter=',',dtype=np.float32)
    else:
      down_frame = np.zeros(center_frame.shape)

  comb_frames = np.concatenate((up_frame[:,:,np.newaxis],center_frame[:,:,np.newaxis],down_frame[:,:,np.newaxis]),axis=2)
  comb_frames[comb_frames<PIX_LOW] = PIX_LOW
  comb_frames[comb_frames>PIX_HIGH] = PIX_HIGH
  comb_frames = MAX_INT * (comb_frames - PIX_LOW) / (PIX_HIGH-PIX_LOW)
  img = comb_frames.astype(np.uint8) 

  output_path = os.path.join(output_dir,f[:-4]+'.png')
  
  if use_gt:
    gname = f.replace('_Im_','_Gt_')
    gt_path = os.path.join(output_dir,gname[:-4]+'.png')

  #Save by PIL
  comb_img = Image.fromarray(img)
  comb_img.save(output_path)
  
  if use_gt:
    gt_img = Image.fromarray(gt_frame)
    gt_img.save(gt_path)
  
  # var_list.append( np.sum(img[:,:,1] != bg_value) * np.var(img[img[:,:,1] != bg_value,1]).astype(np.float64))
  # var_cnt.append(np.sum(img[:,:,1] != bg_value))
  # valid_fnames.append(f)
  var_list = np.sum(img[:,:,1] != bg_value) * np.var(img[img[:,:,1] != bg_value,1]).astype(np.float64)
  var_cnt = np.sum(img[:,:,1] != bg_value)

  return (var_list,var_cnt,f)

def convert_csv2png_main():
  path_to_csv_dir = '/data/20240524_unlabeled_grant_data/NEW_CSVFILES_EMILY_FOLLOWUP_CTs'
  path_to_image_dir = '/data/20240524_unlabeled_grant_data/images'
  data_dir = os.path.join(path_to_csv_dir)
  output_dir = os.path.join(path_to_image_dir)
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  valid_fnames = []
  file_names = [f for f in os.listdir(data_dir) if '.csv' in f and '_Im_' in f]
  file_names.sort()
  with Pool(num_proc) as pool:
    comb_vars = pool.map(partial(convert_csv2png, data_dir=path_to_csv_dir), file_names)
    # comb_vars = pool.map(quick_convert_csv2png,file_names)

  comb_vars = [x for x in zip(*comb_vars)]
  var_list = [x for x in comb_vars[0] if x]
  var_cnt = [x for x in comb_vars[1] if x]
  valid_fnames = [x for x in comb_vars[2] if x]

  #Compute statistics
  json_dict = {}
  mean_val = bg_value/255.
  std_val = np.sqrt(np.sum(var_list)/np.sum(var_cnt))/255.
  json_dict['mean'] = mean_val
  json_dict['std'] = std_val
  with open('{}/info.json'.format(output_dir),'w') as fid:
    json.dump(json_dict,fid)


def convert_sandwich_images(x,output_dir,dataset):
  """
  Convert the 3 channel sandwhich images to 1-channel grayscale
  """
  print(x)
  im = Image.open(osp.join('new_data',dataset,x))
  img = np.array(im)   
  center = np.tile(img[:,:,1][:,:,np.newaxis],(1,1,3))
  cent = Image.fromarray(center)
  cent.save(osp.join(output_dir,dataset,x))

# if __name__ == '__main__':
def convert_sandwich_images_main():
  output_dir = './raw_imgs'
  dataset = 'earthsea'
  img_files = [x for x in os.listdir(osp.join('new_data',dataset)) if '.png' in x and '_Im_' in x]
  if not osp.exists(osp.join(output_dir,dataset)):
    os.mkdir(osp.join(output_dir,dataset)) 
  
  img_files = [x for x in img_files if not osp.exists(osp.join(output_dir,dataset,x))]
  with Pool(num_proc) as pool:
    pool.map(partial(convert_sandwich_images,output_dir=output_dir,dataset=dataset),img_files)

def add_gt_image(x,output_dir,dataset):
  """
  Convert the 3 channel sandwhich images to 1-channel grayscale
  """
  print(x)
  im = Image.open(osp.join('new_data',dataset,x))
  gt = Image.open(osp.join('new_data',dataset,x.replace('_Im_','_Gt_')))
  img = np.array(im,dtype=np.float32)   
  gt = np.array(gt,dtype=np.float32)

  img = np.tile(img[:,:,1][:,:,np.newaxis],(1,1,3))
  img[:,:,0] = img[:,:,0] + gt * 80.
  img = np.minimum(255.,img).astype(np.uint8)
  cent = Image.fromarray(img)
  cent.save(osp.join(output_dir,dataset,x))

# if __name__ == '__main__':
def add_gt_images_main():
  output_dir = '../raw_gts'
  dataset = 'bermuda'
  img_files = [x for x in os.listdir(osp.join('new_data',dataset)) if '.png' in x and '_Im_' in x]
  if not osp.exists(osp.join(output_dir,dataset)):
    os.mkdir(osp.join(output_dir,dataset)) 
  
  img_files = [x for x in img_files if not osp.exists(osp.join(output_dir,dataset,x))]
  with Pool(num_proc) as pool:
    pool.map(partial(add_gt_image,output_dir=output_dir,dataset=dataset),img_files)

def sparsify_masks(x,output_dir):
  """
  Sparsify masks
  """
  thr = 1e-3
  loaded_mask = np.empty(0)
  try: 
    mask = np.load(join(output_dir,x))
  except:
    print("Mask read failed. Pass {}".format(x))
    return
  
  mask[mask<thr] = 0.
  mask = mask.reshape(-1,mask.shape[-1])
  sparse_matrix = scipy.sparse.coo_matrix(mask)
  while not np.array_equal(loaded_mask,mask):
    scipy.sparse.save_npz(join(output_dir,x.replace('.npy','.npz')), sparse_matrix)
    loaded_mask = scipy.sparse.load_npz(join(output_dir,x.replace('.npy','.npz'))).todense()
  
  os.remove(join(output_dir,x))
  print('Finished and removed {}'.format(x))

# if __name__ == '__main__':
def sparsify_masks_main():
  split_num = 4
  # output_dir = 'features/al_{:d}al_e{}_bermuda_trainval/sw3_epoch_800'.format(split_num,'{}')
  # output_dir = 'features/0209_full_bermuda_e{}/sw3_epoch_800'.format('{}')
  # output_dir = 'features/0209_full_bermuda_e{}_cetialpha-darkcity_test/sw3_epoch_800'.format('{}')
  # output_dir = 'features/al_{:d}_e{}_bermuda_trainval/sw3_epoch_800'.format(split_num,'{}')
  # output_dir = 'features/al_2_0219_e2_e{}_bermuda_trainval_al_2_0221diff/sw3_epoch_800'.format('{}')
  # output_dir = 'features/al_{}_repr2_rand_e{}_bermuda_trainval/sw3_epoch_800'.format(split_num,'{}')
  output_dir = 'features/al_{}_mask_addtime1002_e{}_bermuda_trainval_al_4_0224diff/sw3_epoch_800'.format(split_num,'{}')
  # output_dir = 'features/trainval_al_2_maskal_add1256_e{}_bermuda_trainval_al_2_0221diff/sw3_epoch_800'
  num_runs = 4
  
  while True:
    for n in range(num_runs):
      img_files = [x for x in os.listdir(output_dir.format(n)) if '.npy' in x and '_Im_' in x]
      if len(img_files)>0:
        print('Exp {:d}, {:d} Masks to Sparsify'.format(n,len(img_files)))
        with Pool(num_proc) as pool:
          pool.map(partial(sparsify_masks,output_dir=output_dir.format(n)),img_files)

def check_gt(x,output_dir):
  print(x)
  gt = np.array(Image.open(join(output_dir,x)))
  return np.any(gt)

def check_gt_main():
  # dataname = 'bermuda-darkcity'
  path_to_your_gt_txt = '/export/data/sugar_test/png_files/sugar_test_gt_labels.txt'
  output_dir = '/export/data/sugar_test/png_files' #Path to directory that contains your label images.
  output_file_name = '/export/data/sugar_test/png_files/sugar_test_gt_labels.txt' # path to output gt label txt file.
  file_names = [x.strip() for x in open(path_to_your_gt_txt).readlines()]
  
  with Pool(num_proc) as pool:
    labels = pool.map(partial(check_gt,output_dir=output_dir),file_names)
  
  # np.save(join('new_data',dataname,label_file_name.replace('.txt','.npy')),np.array(labels))
  np.savetxt(output_file_name,np.array(labels),delimiter=',')

def median_check_noisy(imgname,data_dir):
  img = np.array(Image.open(os.path.join(data_dir,imgname)))[:,:,1]
  med_img = median(img,disk(1))
  diff_img = img - med_img
  ratio = np.sum(diff_img**2) / (np.sum(img != img[0,0]) + 0.01)
  return ratio 

def check_noisy(imgname,data_dir='./new_data/cetialpha'):
  img = np.array(Image.open(os.path.join(data_dir,imgname)))[:,:,1]
  fft_img = np.fft.fftshift(np.fft.fft2(img))
  mag_fft = np.abs(fft_img)
  low_freq = np.sum(mag_fft[150:350,150:350])
  high_freq = np.sum(mag_fft) - low_freq
  ratio = high_freq / low_freq
  return ratio 

# if __name__ == '__main__':
def check_noisy_main():
  from tqdm import tqdm
  from itertools import groupby
  from operator import itemgetter
  dataset = 'cetialpha-darkcity'
  img_files = [x for x in os.listdir(osp.join('new_data',dataset)) if '.png' in x and '_Im_' in x]
  pro_bar = tqdm(total=len(img_files)) 
  img_ratios = []
  with Pool(num_proc) as pool:
    # for ratio in pool.imap(partial(check_noisy,data_dir=osp.join('new_data',dataset)),img_files):
    for ratio in pool.imap(partial(median_check_noisy,data_dir=osp.join('new_data',dataset)),img_files):
      img_ratios.append(ratio)
      pro_bar.update()
  
  img_file_ratios = zip(img_files,img_ratios)
  get_stack_name = lambda x:x[0].split('_')[-4]
  stack_scores = []
  for k,g in groupby(sorted(img_file_ratios,key=itemgetter(0)),key=get_stack_name):
    glist = list(g)
    stack_files,stack_ratios = zip(*glist)
    stack_scores.append((k,np.median(stack_ratios)))
 
  score_thr = 30.0
  __, scores = zip(*stack_scores)
  # hist, hist_bins = np.histogram(scores,bins=np.arange(0,8,0.5))
  hist, hist_bins = np.histogram(scores,bins=np.arange(0,200,10))

  hf_stacks = [k for k in stack_scores if k[1]>score_thr]
  hf_stacks = sorted(hf_stacks,key=itemgetter(1))[::-1]
  with open('./cache/{}_median_shot_noise_stacks.txt'.format(dataset),'w') as fid:
  # with open('./cache/{}_shot_noise_stacks.txt'.format(dataset),'w') as fid:
    for s in hf_stacks:
      fid.write('{} {:.3f}\n'.format(s[0],s[1]))

def paste_patches(score,sw_ratio):
  h = w = 512
  if len(score.shape) == 4:
    paste_4d = True
  
  elif len(score.shape) == 3:
    paste_4d = False
  
  if paste_4d:
    score_pred = np.zeros((score.shape[0],h,w))
    score_count = np.zeros((score.shape[0],h,w))
  else:
    score_pred = np.zeros((h,w))
    score_count = np.zeros((h,w))

  window_size = 160
  num_w = (sw_ratio*w + window_size-1) // window_size
  num_h = (sw_ratio*h + window_size-1)// window_size
  w_vect = np.linspace(0,w-window_size,num_w).astype(int)
  h_vect = np.linspace(0,h-window_size,num_h).astype(int)
  counter = 0
  #Paste the masks to the full image
  for w_val in w_vect:
      for h_val in h_vect:
          if paste_4d:
            score_pred[:,h_val:h_val+window_size,w_val:w_val+window_size] += score[:,counter,:,:]
            score_count[:,h_val:h_val+window_size,w_val:w_val+window_size] += 1
          else:
            score_pred[h_val:h_val+window_size,w_val:w_val+window_size] += score[counter,:,:]
            score_count[h_val:h_val+window_size,w_val:w_val+window_size] += 1

          counter+=1

  mask = score_pred / score_count

  return mask

def get_gt_stats(x,output_dir):
  gt = np.array(Image.open(join(output_dir,x)))
  score_thr = 0.
  bin_mask = (gt>score_thr).astype(np.uint8)
  boundary_length = perimeter(bin_mask,neighbourhood=4,use_dilation=True)
  labeled_map = label(bin_mask,neighbors=8) 
  num_cc = np.amax(labeled_map)
  print(x,boundary_length,num_cc)
  return (boundary_length,num_cc) 

def get_mask_stats(x,data_dir,num_runs,exp_id,score_thr):
  # print(x)
  # if exp_id is None:
  exp_ids = range(num_runs)
  
  mask_list = []
  for eid in exp_ids:
    if '.npz' in x:
      mask = np.array(scipy.sparse.load_npz(join(data_dir.format(eid),x)).todense())
      mask = mask.reshape(-1,160,160)
      
    else:
      mask = np.load(join(data_dir.format(eid),x))

    mask_list.append(mask)
  
  if exp_id == 'avg':
    mask = np.mean(mask_list,0)
    if mask.shape[0] == 100:
      full_mask = paste_patches(mask,3.0)

    else:
      full_mask = paste_patches(mask,1.0)
    
    blengths = []
    nums = []
    for this_thr in score_thr:
      bin_mask = (full_mask>this_thr).astype(np.uint8)
      boundary_length = perimeter(bin_mask,neighbourhood=4,use_dilation=True)
      labeled_map = label(bin_mask,neighbors=8) 
      num_cc = np.amax(labeled_map)

      blengths.append(boundary_length)
      nums.append(num_cc)
    
    avg_blength = np.mean(blengths)
    avg_num = np.mean(nums)

    return (avg_blength,avg_num) 
  
  elif exp_id == 'all':
    avg_list = [] 
    for mask in mask_list:
      if mask.shape[0] == 100:
        full_mask = paste_patches(mask,3.0)

      else:
        full_mask = paste_patches(mask,1.0)
      
      blengths = []
      nums = []
      for this_thr in score_thr:
        bin_mask = (full_mask>this_thr).astype(np.uint8)
        boundary_length = perimeter(bin_mask,neighbourhood=4,use_dilation=True)
        labeled_map = label(bin_mask,neighbors=8) 
        num_cc = np.amax(labeled_map)

        blengths.append(boundary_length)
        nums.append(num_cc)
      
      avg_blength = np.mean(blengths)
      avg_num = np.mean(nums)
      avg_list.append((avg_blength,avg_num))

    return avg_list

# if __name__ == '__main__':
def compute_gt_stat_main():
  from itertools import groupby
  from operator import itemgetter
  #Cetialpha
  # dataname = 'cetialpha-darkcity'
  # exp_name = '0209_full_bermuda'
  # dataname = 'bermuda'
  # exp_name = '0117_full'
  dataname = 'bermuda'
  # exp_name = 'al_2_0219_e2'
  exp_name = 'al_4'
  num_runs = 4
  ext = '.npz'
  mask_thr = np.arange(0.1,1.0,0.1)
  
  #Use gt or not
  use_gt = False
  if 'cetialpha' in dataname:
    use_gt = False
  #ALL
  # use_exp_id = 'all'
  # use_split = True
  # root_name = 'split_{:d}'

  #AVG
  use_exp_id = 'avg'
  use_split = False
  root_name = 'avgmask'
 
  # use_split = False
  # root_name = 'gtmask'
  if use_gt:
    label_file_name = 'trainval_labels.txt'
    output_dir = join('new_data',dataname)
    file_names = [x.strip() for x in open(join('new_data',dataname,label_file_name)).readlines()]
    mask_thr=0
    file_names.sort()
    with Pool(num_proc) as pool:
      gt_stats = pool.map(partial(get_gt_stats,output_dir=output_dir),file_names)

  else:
    mask_nums = []
    if 'cetialpha' in dataname:
      label_file_name = 'test_{}.txt'.format(exp_name)
      data_dir = 'features/{}_e{}_{}_test/sw3_epoch_800'.format(exp_name,'{}',dataname)
    else:
      label_file_name = 'trainval_{}.txt'.format(exp_name)
      data_dir = 'features/{}_e{}_{}_trainval_al_4_0223diff/sw3_epoch_800'.format(exp_name,'{}',dataname)

    for exp_id in range(num_runs):
      # feature_dir = 'features/{}_e{:d}_{}_test/sw3_epoch_800'.format(exp_name,exp_id,dataname)
      feature_dir = data_dir.format(exp_id)
      mask_files = [x for x in os.listdir(feature_dir) if ext in x]
      mask_files.sort()
      print('Exp {:d}, Masks #{:d}'.format(exp_id,len(mask_files)))
      mask_nums.append(len(mask_files))
    
    file_names = mask_files
    assert(np.all(np.array(mask_nums)==mask_nums[0]))

    pro_bar = tqdm(total=len(mask_files)) 
    gt_stats = []
    with Pool(num_proc) as pool:
      for this_gt_stat in  pool.imap(partial(get_mask_stats,exp_id=use_exp_id,data_dir=data_dir,num_runs=num_runs,score_thr=mask_thr),mask_files):
        gt_stats.append(this_gt_stat)
        pro_bar.update()
  
  stat_name = root_name
  with open(os.path.join('new_data',dataname,label_file_name.replace('.txt','_{}_gt_stats.txt'.format(stat_name))),'w') as fid:
    for g,gstat in enumerate(gt_stats):
      try:
        fid.write('{} {:.3f} {:.3f}\n'.format(file_names[g],gstat[0],gstat[1]))
      except:
        pdb.set_trace()
  
  boundary_lengths, num_ccs = zip(*gt_stats) 
  comb_gt_stats = zip([x.split('_')[-4] for x in file_names],boundary_lengths,num_ccs)
  with open(os.path.join('new_data',dataname,label_file_name.replace('.txt','_{}_gtstack_stats.txt'.format(stat_name))),'w') as fid:
    fid.write('StackName BoundaryLength NumCC\n')
    
    for k,g in groupby(comb_gt_stats,key=itemgetter(0)):
      glist = list(g)
      sum_length = np.sum([x[1] for x in glist])
      sum_cc = np.sum([x[2] for x in glist])
      fid.write('{} {:.3f} {:.3f}\n'.format(k,sum_length,sum_cc))
 
if __name__ == '__main__':
  convert_csv2png_main() 
