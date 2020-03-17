import os
import sys
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='apply masks on disparity map')
parser.add_argument('--disparity_folder', type=str, default='', help='path to the disparity map folder', required=True)
parser.add_argument('--seg_mask_folder', type=str, default='', help='path to the segmentation mask', required=True)
parser.add_argument('--save_dir', type=str, default='', help='path to save your results', required=True)
args = parser.parse_args()

disp_dir = args.disparity_folder
mask_dir = args.seg_mask_folder
disps = os.listdir(disp_dir)
masks = os.listdir(mask_dir)

if(len(disps) == len(masks)):
    print(f'There are {len(disps)} images to be processed.')

disps.sort()
masks.sort()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for disp_id, disp_name in enumerate(disps):
    # load disparity map (.npy)
    disp_path = os.path.join(disp_dir, disp_name)
    print(disp_path)
    disp = np.load(disp_path)
    
    # load corresponding mask
    mask_dir = os.path.join(mask_dir, masks[disp_id])
    print(mask_dir)
    mask = Image.open(mask_dir)
    mask = np.array(mask).astype(float) # turn Image -> numpy array
    mask[mask > 0.0] = 1.0 # turn anything greater than 0 to 1

    masked_disp = np.multiply(mask, disp) # apply masking
    save_path = args.save_dir + str(disp_id).zfill(6) + '.npy'
    np.save(save_path, masked_disp)

    print('%04d/%04d: Masking done.' % (disp_id + 1, len(disps)))

print('Results saved.')