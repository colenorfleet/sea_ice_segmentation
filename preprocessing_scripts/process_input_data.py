import os
import numpy as np
import cv2
import shutil
from tqdm import tqdm


dataset_path = '/home/cole/Documents/NTNU/big_dataset_june24/cole_dataset'
output_path = '/home/cole/Documents/NTNU/datasets'

img_size = 448

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

os.makedirs(os.path.join(output_path, 'raw'))
os.makedirs(os.path.join(output_path, 'raw', 'images'))
os.makedirs(os.path.join(output_path, 'raw', 'ice_masks'))
os.makedirs(os.path.join(output_path, 'raw', 'lidar_masks'))

os.makedirs(os.path.join(output_path, 'morph'))
os.makedirs(os.path.join(output_path, 'morph', 'images'))
os.makedirs(os.path.join(output_path, 'morph', 'ice_masks'))
os.makedirs(os.path.join(output_path, 'morph', 'lidar_masks'))

os.makedirs(os.path.join(output_path, 'otsu'))
os.makedirs(os.path.join(output_path, 'otsu', 'images'))
os.makedirs(os.path.join(output_path, 'otsu', 'ice_masks'))
os.makedirs(os.path.join(output_path, 'otsu', 'lidar_masks'))



### Process mask and convert to binary
for i in tqdm(range(len(topo_files))):
    topo = cv2.imread(os.path.join(dataset_path, 'topo', topo_files[i]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(dataset_path, 'mask', mask_files[i]), cv2.IMREAD_GRAYSCALE)
    real = cv2.imread(os.path.join(dataset_path, 'real', real_files[i]), cv2.IMREAD_COLOR)
    real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)

    assert topo_files[i] == mask_files[i] == real_files[i], 'Files names are not the same'
    filename = topo_files[i][:-4]
    
    # Convert lidar mask to binary
    mask = np.where(mask > 0, 1, 0).astype('uint8')

    # Create hybrid ice mask -- should be a function
    binary_topo_mask = np.where(topo > 0, 1, 0)

    thresholded_gray_image = cv2.threshold(real_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary_otsu_mask = np.where(thresholded_gray_image > 0, 1, 0)

    binary_ice_mask = np.where((binary_topo_mask + binary_otsu_mask) > 1, 1, 0)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    final_binary_ice_mask = cv2.morphologyEx(binary_ice_mask.astype('uint8'), cv2.MORPH_CLOSE, close_kernel, iterations=1)

    closed_binary_topo_mask = cv2.morphologyEx(binary_topo_mask.astype('uint8'), cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Resize real and mask
    real = cv2.resize(real, (img_size, img_size))
    # topo = cv2.resize(topo, (img_size, img_size))
    mask = cv2.resize(mask, (img_size, img_size))

    binary_topo_mask = cv2.resize(binary_topo_mask.astype('float32'), (img_size, img_size))
    closed_binary_topo_mask = cv2.resize(closed_binary_topo_mask, (img_size, img_size))
    final_binary_ice_mask = cv2.resize(final_binary_ice_mask, (img_size, img_size))
    

    # Save images

    # raw
    cv2.imwrite(os.path.join(output_path, 'raw', 'images', filename + '.jpg'), real)
    cv2.imwrite(os.path.join(output_path, 'raw', 'lidar_masks', filename + '.png'), mask*255)
    cv2.imwrite(os.path.join(output_path, 'raw', 'ice_masks', filename + '.png'), binary_topo_mask*255)

    # morph
    cv2.imwrite(os.path.join(output_path, 'morph', 'images', filename + '.jpg'), real)
    cv2.imwrite(os.path.join(output_path, 'morph', 'lidar_masks', filename + '.png'), mask*255)
    cv2.imwrite(os.path.join(output_path, 'morph', 'ice_masks', filename + '.png'), closed_binary_topo_mask*255)

    # otsu
    cv2.imwrite(os.path.join(output_path, 'otsu', 'images', filename + '.jpg'), real)
    cv2.imwrite(os.path.join(output_path, 'otsu', 'lidar_masks', filename + '.png'), mask*255)
    cv2.imwrite(os.path.join(output_path, 'otsu', 'ice_masks', filename + '.png'), final_binary_ice_mask*255)
