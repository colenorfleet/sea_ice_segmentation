
import os
import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

### Find number of negative and positive pixels to figure class imbalance
# INCLUDE FOV MASK? I guess the model is looking at those negative pixels anyway... no its not, loss is masked

output_dir = '/home/cole/Pictures/thesis_report/class_imbalance'

for which_dataset in ['raw', 'morph', 'otsu', 'goNorth', 'roboflow']:


    if which_dataset == 'goNorth':
        label_dir = '/home/cole/Documents/NTNU/datasets/labelled/' + which_dataset + '/ice_masks/'
        mask_dir = '/home/cole/Documents/NTNU/datasets/labelled/' + which_dataset + '/lidar_masks/'
    elif which_dataset == 'roboflow':
        label_dir = '/home/cole/Documents/NTNU/datasets/labelled/' + which_dataset + '/ice_masks/'
        mask_dir = '/home/cole/Documents/NTNU/datasets/labelled/' + which_dataset + '/lidar_masks/'
    else:
        label_dir = '/home/cole/Documents/NTNU/datasets/' + which_dataset + '/ice_masks/'
        mask_dir = '/home/cole/Documents/NTNU/datasets/lidar_masks/'
    


    label_files = sorted(os.listdir(label_dir))
    total_neg_pixels = 0
    total_pos_pixels = 0
    total_nan_pixels = 0
    total_mask_pixels = 0

    for i in range(len(label_files)):
        
        label = cv2.imread(os.path.join(label_dir, label_files[i]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, label_files[i]), cv2.IMREAD_GRAYSCALE)

        label = np.where(label > 0, 1, 0)
        mask = np.where(mask > 0, 1, np.nan)


        label = label*mask

        neg_pixels = np.count_nonzero(label == 0)
        pos_pixels = np.count_nonzero(label == 1)
        nan_pixels = np.count_nonzero(np.isnan(mask))
        mask_pixels = pos_pixels + neg_pixels
        

        assert sum([neg_pixels, pos_pixels, nan_pixels]) == label.size, 'Pixel count is off'

        total_nan_pixels += nan_pixels
        total_neg_pixels += neg_pixels
        total_pos_pixels += pos_pixels
        total_mask_pixels += mask_pixels


    pos_weight = total_neg_pixels / total_pos_pixels
    pos_weight_w_nan = total_neg_pixels / (total_pos_pixels + total_nan_pixels)


    avg_neg_pixels = int(total_neg_pixels / len(label_files))
    avg_pos_pixels = int(total_pos_pixels / len(label_files))
    avg_nan_pixels = int(total_nan_pixels / len(label_files))
    avg_mask_pixels = int(total_mask_pixels / len(label_files))

    
    plt.pie([total_neg_pixels, total_pos_pixels], labels=['Water', 'Ice'], autopct='%1.1f%%', textprops={'fontsize': 18})
    plt.savefig(os.path.join(output_dir, f'{which_dataset}_class_imbalance.png'))
    plt.title(f'{which_dataset} Dataset')
    plt.close()

    print(f'{which_dataset},{np.round(avg_pos_pixels/avg_mask_pixels, 2)},{np.round(avg_neg_pixels/avg_mask_pixels, 2)}')
    
    #print(f"Average negative pixels per image: {avg_neg_pixels}")
    #print(f"Average positive pixels per image: {avg_pos_pixels}")
    #print(f"Average nan pixels in {which_dataset}: {avg_nan_pixels}")
    
    
    #print(f"Positive weight in {which_dataset}: {round(pos_weight, 2)}")
    #print(f"Positive weight with nan in {which_dataset}: {round(pos_weight_w_nan, 2)}")





