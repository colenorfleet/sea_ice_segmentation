from glob import glob
import numpy as np
import random
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


images = sorted(glob('./images/*.jpg'))
lidars = sorted(glob('./annotations/*.png'))


for i in tqdm(range(len(lidars))):

    file_name = images[i].split('/')[-1][:-4]

    image_sample = cv2.imread(images[i])
    topo_sample = cv2.imread(lidars[i], cv2.IMREAD_GRAYSCALE)
    image_gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY)

    # Create a binary mask from the topo image
    binary_topo_mask = np.where(topo_sample > 0, 1, 0)

    # Threshold the real image
    thresholded_gray_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary_otsu_mask = np.where(thresholded_gray_image > 0, 1, 0)

    # Combine the masks
    total_topo_mask = np.where((binary_topo_mask + binary_otsu_mask) > 1, 1, 0)

    # Close the total mask
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    total_mask_close = cv2.morphologyEx(total_topo_mask.astype('uint8'), cv2.MORPH_CLOSE, close_kernel, iterations=1)

    cv2.imwrite(f'./masks/{file_name}.png', total_mask_close*255)
