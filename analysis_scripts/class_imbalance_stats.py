
import os
import cv2
import numpy as np
from tqdm.auto import tqdm

### Find number of negative and positive pixels to figure class imbalance


# INCLUDE FOV MASK? I guess the model is looking at those negative pixels too...

for which_dataset in ['raw', 'morph', 'otsu']:


    label_dir = '/home/cole/Documents/NTNU/datasets/' + which_dataset + '/ice_masks/'
    mask_dir = '/home/cole/Documents/NTNU/datasets/' + which_dataset + '/lidar_masks/'


    label_files = sorted(os.listdir(label_dir))
    total_neg_pixels = 0
    total_pos_pixels = 0

    for i in tqdm(range(len(label_files))):
        label = cv2.imread(os.path.join(label_dir, label_files[i]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, label_files[i]), cv2.IMREAD_GRAYSCALE)

        label = np.where(label > 0, 1, 0)
        mask = np.where(mask > 0, 1, np.nan)

        label = label * mask

        neg_pixels = np.count_nonzero(label == 0)
        pos_pixels = np.count_nonzero(label == 1)
        nan_pixels = np.count_nonzero(np.isnan(label))
        

        assert sum([neg_pixels, pos_pixels, nan_pixels]) == label.size, 'Pixel count is off'

        neg_pixels += nan_pixels
        total_neg_pixels += neg_pixels
        total_pos_pixels += pos_pixels


    pos_weight = total_neg_pixels / total_pos_pixels
    print(f"Total negative pixels in {which_dataset}: {total_neg_pixels}")
    print(f"Total positive pixels in {which_dataset}: {total_pos_pixels}")
    print(f"Positive weight in {which_dataset}: {pos_weight}")



