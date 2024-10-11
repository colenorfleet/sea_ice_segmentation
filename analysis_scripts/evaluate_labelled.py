import sys
sys.path.insert(0, '/home/cole/Documents/NTNU/sea_ice_segmentation')

import os
import cv2
import csv
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from utils.lossfn_utils import calculate_metrics_numpy, calc_SIC_np
from utils.plotting_utils import plot_pixel_classification


base_dir = '/home/cole/Documents/NTNU'
image_dir = base_dir + '/datasets/images'
label_dir = base_dir + '/labelled/masks'
raw_dir = base_dir + '/datasets/raw/ice_masks'
morph_dir = base_dir + '/datasets/morph/ice_masks'
otsu_dir = base_dir + '/datasets/otsu/ice_masks'
lidar_dir = base_dir + '/datasets/lidar_masks'

output_dir = '/home/cole/Pictures/thesis_report/labelled_evaluation'

# overlay_lidar = np.array([255, 0, 0]) * cropped_lidar
# overlay_lidar = overlay_lidar.astype(np.uint8)
# real_lidar_overlay = cv2.addWeighted(np.array(cropped_real), 1, overlay_lidar, 0.5, 0)

label_files = os.listdir(label_dir)
labelled_dict = {0:'water', 1:'Sky', 2:'ice'}

csv_file = os.path.abspath(os.path.join(output_dir, "labelled_metrics.csv"))
csv_header = [
    "Dataset",
    "Image",
    "IOU",
    "DICE",
    "Pixel Accuracy",
    "Precision",
    "Recall",
    "Number True Positive",
    "Number False Positive",
    "Number True Negative",
    "Number False Negative",
    "SIC Manual",
    "SIC Processed",
]

### Logging metrics 
with open(csv_file, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

    for dataset in ['raw', 'morph', 'otsu']:
        for label_file in label_files:
            manual_label = cv2.imread(os.path.join(label_dir, label_file), cv2.IMREAD_GRAYSCALE)
            lidar_mask = cv2.imread(os.path.join(lidar_dir, label_file), cv2.IMREAD_GRAYSCALE)

            manual_label = np.where(manual_label==2, 1, 0)
            lidar_mask = np.where(lidar_mask>0, 1, 0)
            sic_manual = calc_SIC_np(manual_label, lidar_mask)

        
            processed_label = cv2.imread(os.path.join(base_dir, f'datasets/{dataset}/ice_masks', label_file), cv2.IMREAD_GRAYSCALE)
            processed_label = np.where(processed_label>0, 1, 0)
            sic_processed = calc_SIC_np(processed_label, lidar_mask)
            
            iou, dice, pixel_accuracy, precision, recall, num_TP, num_FP, num_TN, num_FN = calculate_metrics_numpy(processed_label, manual_label, lidar_mask)

            csv_writer.writerow(
                [dataset,
                 label_file[:-4],
                 iou,
                 dice,
                 pixel_accuracy,
                 precision,
                 recall,
                 num_TP,
                 num_FP,
                 num_TN,
                 num_FN,
                 sic_manual,
                 sic_processed]
                 )


'''
random.shuffle(label_files)
### Visualizing images
for label_file in label_files:
    img = cv2.imread(os.path.join(image_dir, label_file[:-4] + '.jpg'))
    label = cv2.imread(os.path.join(label_dir, label_file), cv2.IMREAD_GRAYSCALE)
    raw_mask = cv2.imread(os.path.join(raw_dir, label_file), cv2.IMREAD_GRAYSCALE)
    morph_mask = cv2.imread(os.path.join(morph_dir, label_file), cv2.IMREAD_GRAYSCALE)
    otsu_mask = cv2.imread(os.path.join(otsu_dir, label_file), cv2.IMREAD_GRAYSCALE)
    lidar_mask = cv2.imread(os.path.join(lidar_dir, label_file), cv2.IMREAD_GRAYSCALE)

    lidar = np.where(lidar_mask>0, 1, 0)
    raw_mask = np.where(raw_mask>0, 1, 0)
    morph_mask = np.where(morph_mask>0, 1, 0)
    otsu_mask = np.where(otsu_mask>0, 1, 0)
    sky = np.where(label==1, 0, 1)

    ice = np.where(label==2, 1, 0)
    #ice_masked = ice * lidar

    iou, dice, pixel_accuracy, precision, recall, num_TP, num_FP, num_TN, num_FN = calculate_metrics_numpy(raw_mask, ice, lidar)
    print(f'IOU: {iou}, Dice: {dice}, Pixel Accuracy: {pixel_accuracy}, Precision: {precision}, Recall: {recall}, TP: {num_TP}, FP: {num_FP}, TN: {num_TN}, FN: {num_FN}')

    pix_class_raw = plot_pixel_classification(ice*lidar, raw_mask*lidar)
    pix_class_morph = plot_pixel_classification(ice*lidar, morph_mask*lidar)
    pix_class_otsu = plot_pixel_classification(ice*lidar, otsu_mask*lidar)

    fig, axs = plt.subplots(3, 4)
    axs[0,0].imshow(img)
    axs[0,0].set_title(f'Image: {label_file}')
    axs[0,0].axis('off')

    axs[0,1].imshow(ice*lidar)
    axs[0,1].set_title('Manually Labelled Ice')
    axs[0,1].axis('off')

    axs[0,2].imshow(raw_mask)
    axs[0,2].set_title('Processed Mask')
    axs[0,2].axis('off')

    axs[0,3].imshow(pix_class_raw)
    axs[0,3].set_title('Pixel Classification')
    axs[0,3].axis('off')

    ##
    
    axs[1,0].imshow(img)
    axs[1,0].axis('off')

    axs[1,1].imshow(ice*lidar)
    axs[1,1].axis('off')

    axs[1,2].imshow(morph_mask)
    axs[1,2].axis('off')

    axs[1,3].imshow(pix_class_morph)
    axs[1,3].axis('off')

    ##

    axs[2,0].imshow(img)
    axs[2,0].axis('off')

    axs[2,1].imshow(ice*lidar)
    axs[2,1].axis('off')

    axs[2,2].imshow(otsu_mask)
    axs[2,2].axis('off')

    axs[2,3].imshow(pix_class_otsu)
    axs[2,3].axis('off')

    plt.show()


    break
'''
