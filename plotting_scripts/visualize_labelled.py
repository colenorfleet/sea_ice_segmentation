import sys
sys.path.insert(0, '/home/cole/Documents/NTNU/sea_ice_segmentation')
import os
import cv2
import random
import pandas as pd
from utils.plotting_utils import create_mask
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# options:
# 1691152659
# 1690301925
# 1691169006
# 1691169016
# csv files
gonorth_csv = pd.read_csv('/home/cole/Pictures/thesis_report/labelled_evaluation/GoNorth_labelled_metrics.csv')
roboflow_csv = pd.read_csv('/home/cole/Pictures/thesis_report/labelled_evaluation/labelled_metrics.csv')
gonorth_csv['Image'] = gonorth_csv['Image'].astype(str)
roboflow_csv['Image'] = roboflow_csv['Image'].astype(str)
gonorth_csv.set_index(['Dataset', 'Image'], inplace=True)
roboflow_csv.set_index(['Dataset', 'Image'], inplace=True)






# find overlap (if any) between labelled datasets

goNorth_dir = '/home/cole/Documents/NTNU/GoNorth2023-labelled_1'
roboflow_dir = '/home/cole/Documents/NTNU/Binary_Sea_Ice_Segmentation.v3i.png-mask-semantic'
og_dataset_dir = '/home/cole/Documents/NTNU/from_Oskar/big_dataset_june24/cole_dataset'
datasets_dir = '/home/cole/Documents/NTNU/datasets'

goNorth_imgs = os.listdir(os.path.join(goNorth_dir, 'images'))
goNorth_masks = os.listdir(os.path.join(goNorth_dir, 'masks'))

roboflow_files = os.listdir(os.path.join(roboflow_dir, 'test'))
roboflow_imgs = [file for file in roboflow_files if file.endswith('.jpg')]
roboflow_masks = [file for file in roboflow_files if file.endswith('.png')]

roboflow_dict= {}
for file in roboflow_imgs:
    roboflow_dict[file.split('-')[0]] = file.split('-')[1]

traj_dict = {}
for file in roboflow_imgs:
    traj_dict[file.split('-')[0]] = file.split('-')[1][0:2]

roboflow_names = [file.split('-')[0] for file in roboflow_imgs]
goNorth_names = [file.split('.')[0] for file in goNorth_imgs]

overlap = set(roboflow_names).intersection(set(goNorth_names))

fig, axs = plt.subplots(2, 5, figsize=(15, 5))

picks = [str(1691152659), str(1690301925), str(1691169006), str(1691169016)]
for img in picks:

    goNorth_img = cv2.imread(os.path.join(goNorth_dir, 'images', img + '.jpg'))
    goNorth_mask = cv2.imread(os.path.join(goNorth_dir, 'masks', img + '.png'), cv2.IMREAD_GRAYSCALE)

    lidar_mask = cv2.imread(os.path.join(og_dataset_dir, 'mask', f'{img}.{traj_dict[img]}.jpg'), cv2.IMREAD_GRAYSCALE)
    gray_image = cv2.imread(os.path.join(og_dataset_dir, 'real', f'{img}.{traj_dict[img]}.jpg'), cv2.IMREAD_GRAYSCALE)
    lidar_mask = np.where(lidar_mask==0, 1, 0)
    topo = cv2.imread(os.path.join(og_dataset_dir, 'topo', f'{img}.{traj_dict[img]}.jpg'), cv2.IMREAD_GRAYSCALE)


    binary_lidar_mask = lidar_mask.astype('uint8')
    raw_mask = create_mask(gray_image, topo, 'raw')
    morph_mask = create_mask(gray_image, topo, 'morph')
    otsu_mask = create_mask(gray_image, topo, 'otsu')

    go_north_label = Image.fromarray(goNorth_mask)
    lidar_mask = Image.fromarray(binary_lidar_mask)

    raw_mask = Image.fromarray(raw_mask)
    morph_mask = Image.fromarray(morph_mask)
    otsu_mask = Image.fromarray(otsu_mask)

    left = 0
    top = 263
    right = 1430
    bottom = 1063

    lidar_crop = lidar_mask.crop((left, top, right, bottom))
    lidar_crop = lidar_crop.resize(go_north_label.size)
    raw_crop = raw_mask.crop((left, top, right, bottom))
    raw_crop = raw_crop.resize(go_north_label.size)
    raw_crop = np.array(raw_crop).astype(np.uint8)
    morph_crop = morph_mask.crop((left, top, right, bottom))
    morph_crop  = morph_crop.resize(go_north_label.size)
    otsu_crop = otsu_mask.crop((left, top, right, bottom))
    otsu_crop = otsu_crop.resize(go_north_label.size)


    overlay_lidar = np.array([0, 0, 255]) * np.stack([np.array(lidar_crop), np.array(lidar_crop), np.array(lidar_crop)], axis=2)
    overlay_lidar = overlay_lidar.astype(np.uint8)
    mask_lidar_overlap = cv2.addWeighted(np.stack([np.array(goNorth_mask)*225, np.array(goNorth_mask)*225, np.array(goNorth_mask)*225], axis=2), 1, overlay_lidar, 1, 0)
    raw_lidar_overlay = cv2.addWeighted(np.stack([raw_crop*255, raw_crop*255, raw_crop*255], axis=2), 1, overlay_lidar, 0.5, 0)
    morph_lidar_overlay = cv2.addWeighted(np.stack([np.array(morph_crop)*255, np.array(morph_crop)*255, np.array(morph_crop)*255], axis=2), 1, overlay_lidar, 0.5, 0)
    otsu_lidar_overlay = cv2.addWeighted(np.stack([np.array(otsu_crop)*255, np.array(otsu_crop)*255, np.array(otsu_crop)*255], axis=2), 1, overlay_lidar, 0.5, 0)



    
    roboflow_img = cv2.imread(os.path.join(roboflow_dir, 'test', f'{img}-{roboflow_dict[img]}'))
    roboflow_mask = cv2.imread(os.path.join(roboflow_dir, 'test', f'{img}-{roboflow_dict[img][:-4]}_mask.png'), cv2.IMREAD_GRAYSCALE)
    roboflow_mask = np.where(roboflow_mask==2, 1, 0).astype(np.uint8)

    lidar_preproc_mask = cv2.imread(os.path.join(datasets_dir, 'lidar_masks', f'{img}.{traj_dict[img]}.png'), cv2.IMREAD_GRAYSCALE)
    raw_proc_mask = cv2.imread(os.path.join(datasets_dir, 'raw/ice_masks', f'{img}.{traj_dict[img]}.png'), cv2.IMREAD_GRAYSCALE)
    morph_proc_mask = cv2.imread(os.path.join(datasets_dir, 'morph/ice_masks', f'{img}.{traj_dict[img]}.png'), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    otsu_proc_mask = cv2.imread(os.path.join(datasets_dir, 'otsu/ice_masks', f'{img}.{traj_dict[img]}.png'), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    lidar_proc_mask = np.where(lidar_preproc_mask==0,1,0).astype(np.uint8)
    raw_proc_mask = np.where(raw_proc_mask>0, 1, 0).astype(np.uint8)
    morph_proc_mask = np.where(morph_proc_mask>0,1,0).astype(np.uint8)
    otsu_proc_mask = np.where(otsu_proc_mask>0,1,0).astype(np.uint8)

    lidar_preproc_mask = np.where(lidar_preproc_mask>0,1,0)
    roboflow_mask = (roboflow_mask*lidar_preproc_mask).astype(np.uint8)

    lidar_proc_mask = np.array([0,0,255]) * np.stack([lidar_proc_mask, lidar_proc_mask, lidar_proc_mask], axis=2)
    lidar_proc_mask = lidar_proc_mask.astype(np.uint8)
    mask_proc_overlay = cv2.addWeighted(np.stack([roboflow_mask*225, roboflow_mask*225, roboflow_mask*225], axis=2), 1, lidar_proc_mask, 0.5, 0)
    raw_proc_overlay = cv2.addWeighted(np.stack([raw_proc_mask*255, raw_proc_mask*255, raw_proc_mask*255], axis=2), 1, lidar_proc_mask, 0.5, 0)
    morph_proc_overlay = cv2.addWeighted(np.stack([morph_proc_mask*255, morph_proc_mask*255, morph_proc_mask*255], axis=2), 1, lidar_proc_mask, 0.5, 0)
    otsu_proc_overlay = cv2.addWeighted(np.stack([otsu_proc_mask*255, otsu_proc_mask*255, otsu_proc_mask*255], axis=2), 1, lidar_proc_mask, 0.5, 0)

    roboflow_img = cv2.resize(roboflow_img, (715, 531))
    roboflow_mask = cv2.resize(mask_proc_overlay, (715, 531))
    raw_proc_mask = cv2.resize(raw_proc_overlay, (715, 531))
    morph_proc_mask = cv2.resize(morph_proc_overlay, (715, 531))
    otsu_proc_mask = cv2.resize(otsu_proc_overlay, (715, 531))

    gn_raw_iou = gonorth_csv.loc[('raw', f'{img}.{traj_dict[img]}'), 'IOU'].round(2)
    gn_morph_iou = gonorth_csv.loc[('morph', f'{img}.{traj_dict[img]}'), 'IOU'].round(2)
    gn_otsu_iou = gonorth_csv.loc[('otsu', f'{img}.{traj_dict[img]}'), 'IOU'].round(2)

    rf_raw_iou = roboflow_csv.loc[('raw', f'{img}.{traj_dict[img]}'), 'IOU'].round(2)
    rf_morph_iou = roboflow_csv.loc[('morph', f'{img}.{traj_dict[img]}'), 'IOU'].round(2)
    rf_otsu_iou = roboflow_csv.loc[('otsu', f'{img}.{traj_dict[img]}'), 'IOU'].round(2)

    print(f"Go North img {img} Raw Mask IoU: {gn_raw_iou}")
    print(f"Go North img {img} Morph Mask IoU: {gn_morph_iou}")
    print(f"Go North img {img} Otsu Mask IoU: {gn_otsu_iou}")

    print(f"Roboflow img {img} Raw Mask IoU: {rf_raw_iou}")
    print(f"Roboflow img {img} Morph Mask IoU: {rf_morph_iou}")
    print(f"Roboflow img {img} Otsu Mask IoU: {rf_otsu_iou}")


    axs[0, 0].imshow(roboflow_img)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(roboflow_mask)
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(raw_proc_mask)
    axs[0, 2].axis('off')

    axs[0, 3].imshow(morph_proc_mask)
    axs[0, 3].axis('off')

    axs[0, 4].imshow(otsu_proc_mask)
    axs[0, 4].axis('off')

    axs[1, 0].imshow(goNorth_img)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(mask_lidar_overlap)
    axs[1, 1].axis('off')

    axs[1, 2].imshow(raw_lidar_overlay)
    axs[1, 2].axis('off')

    axs[1, 3].imshow(morph_lidar_overlay)
    axs[1, 3].axis('off')
    
    axs[1, 4].imshow(otsu_lidar_overlay)
    axs[1, 4].axis('off')

    plt.tight_layout()
    plt.savefig(f'/home/cole/Pictures/thesis_report/labelled_evaluation/picks/{img}_visual.png')
    plt.close()

    img_list = [roboflow_img, roboflow_mask, raw_proc_mask, morph_proc_mask, otsu_proc_mask, goNorth_img, mask_lidar_overlap, raw_lidar_overlay, morph_lidar_overlay, otsu_lidar_overlay]
    for i, image in enumerate(img_list):
        
        cv2.imwrite(f'/home/cole/Pictures/thesis_report/labelled_evaluation/picks/{img}_visual_part{i}.png', image)

    
    

