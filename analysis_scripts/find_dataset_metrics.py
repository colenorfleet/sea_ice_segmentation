

# Find mean and std dev of image dataset

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_mean_stddev(image_dir):
    mean = np.zeros(3)
    std = np.zeros(3)
    num_images = 0

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'):
                num_images += 1
                img = Image.open(os.path.join(root, file))
                img = np.array(img).astype(np.float32) / 255
                mean += np.mean(img, axis=(0, 1))
                std += np.std(img, axis=(0, 1))

    mean /= num_images
    std /= num_images

    return mean, std



image_dir = './images/'

mean, std = get_mean_stddev(image_dir)

print(f'Mean: {mean}')
print(f'Std Dev: {std}')
