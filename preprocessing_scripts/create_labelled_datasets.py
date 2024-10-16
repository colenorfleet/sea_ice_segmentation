
import sys
sys.path.insert(0, '/home/cole/Documents/NTNU/sea_ice_segmentation')
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# find overlap (if any) between labelled datasets
goNorth_dir = '/home/cole/Documents/NTNU/GoNorth2023-labelled_1'
roboflow_dir = '/home/cole/Documents/NTNU/Binary_Sea_Ice_Segmentation.v3i.png-mask-semantic'
og_dataset_dir = '/home/cole/Documents/NTNU/from_Oskar/big_dataset_june24/cole_dataset'
datasets_dir = '/home/cole/Documents/NTNU/datasets'
gonorth_output = '/home/cole/Documents/NTNU/datasets/labelled/goNorth'

goNorth_imgs = os.listdir(os.path.join(goNorth_dir, 'images'))
goNorth_masks = os.listdir(os.path.join(goNorth_dir, 'masks'))

og_imgs = os.listdir(os.path.join(og_dataset_dir, 'real'))
roboflow_files = os.listdir(os.path.join(roboflow_dir, 'test'))

roboflow_imgs = [file for file in roboflow_files if file.endswith('.jpg')]
roboflow_masks = [file for file in roboflow_files if file.endswith('.png')]

roboflow_dict= {}
for file in roboflow_imgs:
    roboflow_dict[file.split('-')[0]] = file.split('-')[1]

traj_dict = {}
for file in og_imgs:
    traj_dict[file.split('.')[0]] = file.split('.')[1]

overlap = set([file.split('.')[0] for file in goNorth_imgs]).intersection(set(traj_dict.keys()))
overlap_roboflow = set(roboflow_dict.keys()).intersection(set(traj_dict.keys()))

img_size = 256

# resize and save roboflow
print('Saving roboflow')
for img in tqdm(overlap_roboflow):
    image = cv2.imread(os.path.join(roboflow_dir, 'test', img + '-' + roboflow_dict[img]))
    mask = cv2.imread(os.path.join(roboflow_dir, 'test', img + '-' + roboflow_dict[img][0:-4] + '_mask.png'), cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask==2, 1, 0).astype('uint8')

    lidar_mask = cv2.imread(os.path.join(datasets_dir, 'lidar_masks', img + '.' + traj_dict[img] + '.png'), cv2.IMREAD_GRAYSCALE)
    lidar_mask = np.where(lidar_mask>0, 1, 0).astype('uint8')

    lidar_mask = cv2.resize(lidar_mask, (img_size,img_size), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (img_size,img_size), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (img_size,img_size), interpolation=cv2.INTER_NEAREST)

    assert np.all((lidar_mask==0) | (lidar_mask==1)), 'lidar is not binary'
    assert np.all((mask==0) | (mask==1)), 'roboflow mask is not binary'

    cv2.imwrite(os.path.join(datasets_dir, 'labelled', 'roboflow', 'images', img + '.' + traj_dict[img] + '.jpg'), image)
    cv2.imwrite(os.path.join(datasets_dir, 'labelled', 'roboflow', 'ice_masks', img + '.' + traj_dict[img] + '.png'), mask*255)
    cv2.imwrite(os.path.join(datasets_dir, 'labelled', 'roboflow', 'lidar_masks', img + '.' + traj_dict[img] + '.png'), lidar_mask*255)



# resize and save goNorth
print('Saving goNorth')
for img in tqdm(overlap):
    image = cv2.imread(os.path.join(goNorth_dir, 'images', img + '.jpg'))
    mask = cv2.imread(os.path.join(goNorth_dir, 'masks', img + '.png'), cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask>0, 1, 0).astype('uint8')

    lidar_mask = cv2.imread(os.path.join(og_dataset_dir, 'mask', img + '.' + traj_dict[img] + '.jpg'), cv2.IMREAD_GRAYSCALE)
    lidar_mask = np.where(lidar_mask>0, 1, 0).astype('uint8')
    lidar_mask = Image.fromarray(lidar_mask)

    left = 0
    top = 263
    right = 1430
    bottom = 1063

    lidar_crop = lidar_mask.crop((left, top, right, bottom))
    lidar_crop = np.array(lidar_crop).astype('uint8')

    lidar_crop = cv2.resize(lidar_crop, (img_size,img_size), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (img_size,img_size), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (img_size,img_size), interpolation=cv2.INTER_NEAREST)

    assert np.all((lidar_crop==0) | (lidar_crop==1)), 'lidar is not binary'
    assert np.all((mask==0) | (mask==1)), 'goNorth mask is not binary'

    cv2.imwrite(os.path.join(gonorth_output, 'images', img + '.jpg'), image)
    cv2.imwrite(os.path.join(gonorth_output, 'ice_masks', img + '.png'), mask*255)
    cv2.imwrite(os.path.join(gonorth_output, 'lidar_masks', img + '.png'), lidar_crop*255)



    





    

    