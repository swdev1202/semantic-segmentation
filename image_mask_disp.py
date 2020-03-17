import os
import sys
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='apply masks on disparity map')
parser.add_argument('--disparity_folder', type=str, default='', help='path to the disparity map folder', required=True)
parser.add_argument('--label_folder', type=str, default='', help='path to the ground truth label files', required=True)
parser.add_argument('--save_dir', type=str, default='', help='path to save your results', required=True)
args = parser.parse_args()

disp_dir = args.disparity_folder
label_dir = args.label_folder
disps = os.listdir(disp_dir)
labels = os.listdir(label_dir)

if(len(disps) == len(labels)):
    print(f'There are {len(disps)} images to be processed.')

disps.sort()
labels.sort()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for disp_id, disp_name in enumerate(disps):
    # load disparity map (.npy)
    disp_dir = os.path.join(disp_dir, disp_name)
    print(disp_dir)
    disp = np.load(disp_dir)

    # create an empty mask
    mask = np.zeros_like(disp)
    mask
    
    # load corresponding label to mask
    label_file = os.path.join(label_dir, labels[disp_id])
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # collect only the car information
    for line in lines:
        annots = line.split(' ')
        if(annots[0] == 'Car'):
            x_min = int(round(float(annots[4])))
            y_min = int(round(float(annots[5])))
            x_max = int(round(float(annots[6])))
            y_max = int(round(float(annots[7])))

            mask[y_min:y_max, x_min:x_max] = 1.0

    
    # save the masked disparity
    masked_disp = np.multiply(mask, disp) # apply masking
    save_path = args.save_dir + str(disp_id).zfill(6) + '.npy'
    np.save(save_path, masked_disp)

    print('%04d/%04d: Masking done.' % (disp_id + 1, len(disps)))

print('Results saved.')