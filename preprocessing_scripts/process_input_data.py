import sys
sys.path.insert(0, '/home/cole/Documents/NTNU/sea_ice_segmentation')

import os
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from utils.plotting_utils import create_mask


dataset_path = '/home/cole/Documents/NTNU/from_Oskar/big_dataset_june24/cole_dataset'
output_path = '/home/cole/Documents/NTNU/datasets'

img_size = 256

assert 'topo' and 'mask' and 'real' in os.listdir(dataset_path), 'topo, mask, or real not in dataset folder'

assert os.listdir(os.path.join(dataset_path, 'topo')) == os.listdir(os.path.join(dataset_path, 'mask')) == os.listdir(os.path.join(dataset_path, 'real')), 'topo, mask, and real directories are not the same'

topo_files = sorted(os.listdir(os.path.join(dataset_path, 'topo')))
mask_files = sorted(os.listdir(os.path.join(dataset_path, 'mask')))
real_files = sorted(os.listdir(os.path.join(dataset_path, 'real')))


# Check if output directories exist, if they do, delete them
if os.path.exists(os.path.join(output_path, 'raw')):
    print('raw directory already exists, deleting')
    shutil.rmtree(os.path.join(output_path, 'raw'))

if os.path.exists(os.path.join(output_path, 'morph')):
    print('morph directory already exists, deleting')
    shutil.rmtree(os.path.join(output_path, 'morph'))

if os.path.exists(os.path.join(output_path, 'otsu')):
    print('otsu directory already exists, deleting')
    shutil.rmtree(os.path.join(output_path, 'otsu'))


os.makedirs(os.path.join(output_path, 'lidar_masks'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

os.makedirs(os.path.join(output_path, 'raw'))
os.makedirs(os.path.join(output_path, 'raw', 'ice_masks'))

os.makedirs(os.path.join(output_path, 'morph'))
os.makedirs(os.path.join(output_path, 'morph', 'ice_masks'))

os.makedirs(os.path.join(output_path, 'otsu'))
os.makedirs(os.path.join(output_path, 'otsu', 'ice_masks'))




### Process mask and convert to binary
for i in tqdm(range(len(topo_files))):
    topo = cv2.imread(os.path.join(dataset_path, 'topo', topo_files[i]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(dataset_path, 'mask', mask_files[i]), cv2.IMREAD_GRAYSCALE)
    real = cv2.imread(os.path.join(dataset_path, 'real', real_files[i]), cv2.IMREAD_COLOR)
    real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    mask = np.where(mask>0, 1, 0).astype('uint8')

    assert topo_files[i] == mask_files[i] == real_files[i], 'Files names are not the same'
    filename = topo_files[i][:-4]

    raw_mask = create_mask(real_gray, topo, 'raw')
    morph_mask = create_mask(real_gray, topo, 'morph')
    otsu_mask = create_mask(real_gray, topo, 'otsu')

    # Resize real and mask
    real = cv2.resize(real, (img_size, img_size))
    mask = cv2.resize(mask, (img_size, img_size))

    raw_mask = cv2.resize(raw_mask, (img_size, img_size))
    morph_mask = cv2.resize(morph_mask, (img_size, img_size))
    otsu_mask = cv2.resize(otsu_mask, (img_size, img_size))

    assert np.all((mask == 0) | (mask == 1)), 'Mask is not binary'
    assert np.all((raw_mask == 0) | (raw_mask == 1)), 'Raw is not binary'
    assert np.all((morph_mask == 0) | (morph_mask == 1)), 'Morph is not binary'
    assert np.all((otsu_mask == 0) | (otsu_mask == 1)), 'Otsu is not binary'
    
    # Save images
    cv2.imwrite(os.path.join(output_path, 'images', filename + '.jpg'), real)
    cv2.imwrite(os.path.join(output_path, 'lidar_masks', filename + '.png'), mask*255)

    # raw
    cv2.imwrite(os.path.join(output_path, 'raw', 'ice_masks', filename + '.png'), raw_mask*255)

    # morph
    cv2.imwrite(os.path.join(output_path, 'morph', 'ice_masks', filename + '.png'), morph_mask*255)

    # otsu
    cv2.imwrite(os.path.join(output_path, 'otsu', 'ice_masks', filename + '.png'), otsu_mask*255)
