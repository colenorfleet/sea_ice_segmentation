
import os
import cv2
import csv
import numpy as np
from tqdm.auto import tqdm


file_split_directory = '/home/cole/Documents/NTNU/datasets/'
output_dir = '/home/cole/Pictures/thesis_report/'
csv_file = os.path.abspath(os.path.join(output_dir, "dataset_split_metrics.csv"))
csv_header = ['Split', '# of Images', 'Raw SIC', 'Morph SIC', 'Otsu SIC']

with open(csv_file, mode='w') as c_file:
    csv_writer = csv.writer(c_file)
    csv_writer.writerow(csv_header)

    for split in ['train', 'val', 'test']:

        total_sic_raw = 0
        total_sic_morph = 0
        total_sic_otsu = 0
        print(f"Calculating SIC for {split} split")

        with open(os.path.join(file_split_directory, f'{split}.txt'), 'r') as f:
            files = f.read().splitlines()

            print(len(files))

            for file in tqdm(files):
                raw_mask = cv2.imread(os.path.join(file_split_directory, 'raw/ice_masks', f'{file}.png'), cv2.IMREAD_GRAYSCALE)
                morph_mask = cv2.imread(os.path.join(file_split_directory, 'morph/ice_masks', f'{file}.png'), cv2.IMREAD_GRAYSCALE)
                otsu_mask = cv2.imread(os.path.join(file_split_directory, 'otsu/ice_masks', f'{file}.png'), cv2.IMREAD_GRAYSCALE)
                lidar_mask = cv2.imread(os.path.join(file_split_directory, 'lidar_masks', f'{file}.png'), cv2.IMREAD_GRAYSCALE)

                assert set(raw_mask.flatten()) == {0} or {0, 255}, f"raw mask {file} is not binary, {set(raw_mask.flatten())}"
                assert set(morph_mask.flatten()) == {0} or {0, 255}, f"morph mask {file} is not binary"
                assert set(otsu_mask.flatten()) == {0} or {0, 255}, f"otsu mask {file} is not binary"
                assert set(lidar_mask.flatten()) == {0, 255}, f"lidar mask {file} is not binary"

                raw_mask = np.where(raw_mask > 0, 1, 0)
                morph_mask = np.where(morph_mask > 0, 1, 0)
                otsu_mask = np.where(otsu_mask > 0, 1, 0)
                lidar_mask = np.where(lidar_mask > 0, 1, np.nan)

                raw_mask = raw_mask * lidar_mask
                morph_mask = morph_mask * lidar_mask
                otsu_mask = otsu_mask * lidar_mask

                raw_pos_pixels = np.count_nonzero(raw_mask == 1)
                raw_neg_pixels = np.count_nonzero(raw_mask == 0)

                morph_pos_pixels = np.count_nonzero(morph_mask == 1)
                morph_neg_pixels = np.count_nonzero(morph_mask == 0)

                otsu_pos_pixels = np.count_nonzero(otsu_mask == 1)
                otsu_neg_pixels = np.count_nonzero(otsu_mask == 0)

                sic_raw = raw_pos_pixels / (raw_pos_pixels + raw_neg_pixels)
                sic_morph = morph_pos_pixels / (morph_pos_pixels + morph_neg_pixels)
                sic_otsu = otsu_pos_pixels / (otsu_pos_pixels + otsu_neg_pixels)

                total_sic_raw += sic_raw
                total_sic_morph += sic_morph
                total_sic_otsu += sic_otsu
        
        avg_sic_raw = total_sic_raw / len(files)
        avg_sic_morph = total_sic_morph / len(files)
        avg_sic_otsu = total_sic_otsu / len(files)

        print(f"Average Raw SIC for {split}: {round(avg_sic_raw, 2)}")
        print(f"Average Morph SIC for {split}: {round(avg_sic_morph, 2)}")
        print(f"Average Otsu SIC for {split}: {round(avg_sic_otsu, 2)}")

        csv_writer.writerow(
                        [
                            split,
                            int(len(files)),
                            round(avg_sic_raw, 2),
                            round(avg_sic_morph, 2),
                            round(avg_sic_otsu, 2)
                        ]
                    )







            

