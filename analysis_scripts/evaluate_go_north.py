import sys
sys.path.insert(0, '/home/cole/Documents/NTNU/sea_ice_segmentation')

import os
import csv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import random
from itertools import combinations
from utils.lossfn_utils import calc_SIC_np, calculate_metrics_numpy
from utils.plotting_utils import plot_pixel_classification, create_mask

dataset_dir = '/home/cole/Documents/NTNU/from_Oskar/big_dataset_june24/cole_dataset'
# dataset_dir = '/home/cole/Documents/NTNU/datasets/images'
go_north_dir = '/home/cole/Documents/NTNU/GoNorth2023-labelled_1'
processed_dataset_dir = '/home/cole/Documents/NTNU/datasets'
output_dir = '/home/cole/Pictures/thesis_report/labelled_evaluation'

go_north_imgs = os.listdir(go_north_dir + '/images')
go_north_imgs = [img.split('.')[0] for img in go_north_imgs]

dataset_files = set(os.listdir(dataset_dir + '/real'))
trajectory_dict = {}
for img in dataset_files:
    trajectory_dict[img.split('.')[0]] = img.split('.')[1]


overlap = set(go_north_imgs).intersection(set(trajectory_dict.keys()))


### Record metrics
csv_file = os.path.abspath(os.path.join(output_dir, "GoNorth_labelled_metrics.csv"))
csv_header = [
    "Dataset",
    "Image",
    "IOU",
    "DICE",
    "Pixel Accuracy",
    "Precision",
    "Recall",
    "Number True Positive",
    "Number Faprocessed_masklse Positive",
    "Number True Negative",
    "Number False Negative",
    "SIC Manual",
    "SIC Processed",
]

with open(csv_file, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

    for dataset in ['raw', 'morph', 'otsu']:
        for img in overlap:

            go_north_label = cv2.imread(os.path.join(go_north_dir, 'masks', f'{img}.png'), cv2.IMREAD_GRAYSCALE)
            lidar_mask = cv2.imread(os.path.join(dataset_dir, 'mask', f'{img}.{trajectory_dict[img]}.jpg'), cv2.IMREAD_GRAYSCALE)
            gray_image = cv2.imread(os.path.join(dataset_dir, 'real', f'{img}.{trajectory_dict[img]}.jpg'), cv2.IMREAD_GRAYSCALE)

            lidar_mask = np.where(lidar_mask>0, 1, 0)

            topo = cv2.imread(os.path.join(dataset_dir, 'topo', f'{img}.{trajectory_dict[img]}.jpg'), cv2.IMREAD_GRAYSCALE)

            binary_lidar_mask = lidar_mask.astype('uint8')

            processed_mask = create_mask(gray_image, topo, dataset)

            go_north_label = Image.fromarray(go_north_label)
            lidar_mask = Image.fromarray(binary_lidar_mask)
            proc_mask = Image.fromarray(processed_mask)

            ## CROP
            ## original image
            left = 0
            top = 263
            right = 1430
            bottom = 1063

            lidar_crop = lidar_mask.crop((left, top, right, bottom))
            lidar_crop = lidar_crop.resize(go_north_label.size)
            mask_crop = proc_mask.crop((left, top, right, bottom))
            mask_crop = mask_crop.resize(go_north_label.size)
            
            assert lidar_crop.size == mask_crop.size == go_north_label.size, "Sizes do not match"

            metrics = calculate_metrics_numpy(np.array(mask_crop), np.array(go_north_label), np.array(lidar_crop))
            
            sic_manual = calc_SIC_np(np.array(go_north_label), np.array(lidar_crop))
            sic_processed = calc_SIC_np(np.array(mask_crop), np.array(lidar_crop))

            csv_writer.writerow(
                [dataset,
                str(f"{img}.{trajectory_dict[img]}"),
                metrics['iou'],
                metrics['dice_score'],
                metrics['pixel_accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['num_TP'],
                metrics['num_FP'],
                metrics['num_TN'],
                metrics['num_FN'],
                sic_manual,
                sic_processed,
                ]
            )

  


            
'''
### Visualize the images 

random.shuffle(go_north_imgs)

for img in go_north_imgs:

    og_image = Image.open(os.path.join(dataset_dir, 'real', f'{img}.{trajectory_dict[img]}.jpg'))
    gray_image = og_image.convert('L')
    og_topo = Image.open(os.path.join(dataset_dir, 'topo', f'{img}.{trajectory_dict[img]}.jpg'))
    gn_image = Image.open(os.path.join(go_north_dir, 'images', f'{img}.jpg'))
    gn_mask = Image.open(os.path.join(go_north_dir, 'masks', f'{img}.png'))
    lidar_mask = Image.open(os.path.join(dataset_dir, 'mask', f'{img}.{trajectory_dict[img]}.jpg'))

    # disregard lidar mask for now
    binary_lidar_mask = np.where(np.array(lidar_mask)==0, 1, 0).astype('uint8')
    binary_topo_mask = np.where(np.array(og_topo) > 0, 1, 0).astype('uint8')
    thresholded_gray_image = cv2.threshold(np.array(gray_image), 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary_otsu_mask = np.where(thresholded_gray_image > 0, 1, 0)
    binary_ice_mask = np.where((binary_topo_mask + binary_otsu_mask) > 1, 1, 0)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    final_binary_ice_mask = cv2.morphologyEx(binary_ice_mask.astype('uint8'), cv2.MORPH_CLOSE, close_kernel, iterations=1)
    closed_binary_topo_mask = cv2.morphologyEx(binary_topo_mask.astype('uint8'), cv2.MORPH_CLOSE, close_kernel, iterations=1)
    
    lidar_mask = Image.fromarray(binary_lidar_mask)
    raw = Image.fromarray(binary_topo_mask)
    morph = Image.fromarray(closed_binary_topo_mask)
    otsu = Image.fromarray(final_binary_ice_mask)

    ## original image
    left = 0
    top = 263
    right = 1430
    bottom = 1063

    cropped_real = og_image.crop((left, top, right, bottom))
    cropped_real = cropped_real.resize(gn_image.size)

    cropped_topo = og_topo.crop((left, top, right, bottom))
    cropped_topo = cropped_topo.resize(gn_image.size)

    cropped_raw = raw.crop((left, top, right, bottom))
    cropped_raw = cropped_raw.resize(gn_image.size)

    cropped_morph = morph.crop((left, top, right, bottom))
    cropped_morph = cropped_morph.resize(gn_image.size)
    cropped_otsu = otsu.crop((left, top, right, bottom))
    cropped_otsu = cropped_otsu.resize(gn_image.size)

    cropped_lidar = lidar_mask.crop((left, top, right, bottom))
    cropped_lidar = cropped_lidar.resize(gn_image.size)


    overlay_lidar = np.array([255, 0, 0]) * cropped_lidar
    overlay_lidar = overlay_lidar.astype(np.uint8)
    real_lidar_overlay = cv2.addWeighted(np.array(cropped_real), 1, overlay_lidar, 0.5, 0)

    gn_extra_mask = np.stack([np.array(gn_mask)]*3, axis=2)
    gn_extra_mask = np.array([255, 0, 0]) * gn_extra_mask
    gn_extra_mask = gn_extra_mask.astype(np.uint8)

    raw_array = np.stack([np.array(cropped_raw)*200]*3, axis=2).astype(np.uint8)
    morph_array = np.stack([np.array(cropped_morph)*200]*3, axis=2).astype(np.uint8)
    otsu_array = np.stack([np.array(cropped_otsu)*200]*3, axis=2).astype(np.uint8)

    gn_mask_raw = cv2.addWeighted(raw_array, 1, gn_extra_mask, 0.5, 0)
    gn_mask_morph = cv2.addWeighted(morph_array, 1, gn_extra_mask, 0.5, 0)
    gn_mask_otsu = cv2.addWeighted(otsu_array, 1, gn_extra_mask, 0.5, 0)

    fig, axs = plt.subplots(3, 4)

    axs[0,0].imshow(gn_image)
    axs[0,0].axis('off')
    axs[0,0].set_title('GoNorth Image')

    axs[0,1].imshow(gn_mask, cmap='gray')
    axs[0,1].axis('off')
    axs[0,1].set_title('GoNorth Mask')

    axs[0,2].imshow(cropped_lidar, cmap='gray')
    axs[0,2].axis('off')
    axs[0,2].set_title('Lidar Mask')

    axs[1,0].imshow(real_lidar_overlay)
    axs[1,0].axis('off')
    axs[1,0].set_title('Cropped Image')

    axs[1,1].imshow(cropped_raw, cmap='gray')
    axs[1,1].axis('off')
    axs[1,1].set_title('Raw Mask')

    axs[1,2].imshow(cropped_morph, cmap='gray')
    axs[1,2].axis('off')
    axs[1,2].set_title('Morph Mask')

    axs[1,3].imshow(cropped_otsu, cmap='gray')
    axs[1,3].axis('off')
    axs[1,3].set_title('Otsu Mask')

    axs[2,0].imshow(cropped_topo, cmap='gray')
    axs[2,0].axis('off')
    axs[2,0].set_title('Cropped Topo')
    
    axs[2,1].imshow(gn_mask_raw)
    axs[2,1].axis('off')
    axs[2,1].set_title('Raw Mask Overlay')

    axs[2,2].imshow(gn_mask_morph)
    axs[2,2].axis('off')
    axs[2,2].set_title('Morph Mask Overlay')

    axs[2,3].imshow(gn_mask_otsu)
    axs[2,3].axis('off')
    axs[2,3].set_title('Otsu Mask Overlay')


    plt.tight_layout()

    plt.show()
    break
    plt.close()





## resized image
    #left = 0
    #top = 125
    #right = 512
    #bottom = 512

'''