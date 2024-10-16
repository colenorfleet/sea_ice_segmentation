import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_path = '/home/cole/Pictures/thesis_report/test_set_statistics/256'

def compare_model_performance(mode='dataset'):

    models = ['unet', 'deeplabv3plus', 'segformer']
    datasets = ['raw', 'morph', 'otsu']
    metrics = ['IOU', 'DICE', 'Pixel Accuracy', 'Precision', 'Recall']

    if mode == 'model':
        for model in models:
            raw_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/raw/evaluation_scores.csv')
            morph_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/morph/evaluation_scores.csv')
            otsu_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/otsu/evaluation_scores.csv')

            for metric in metrics:

                plt.figure(figsize=(20, 10))
                plt.title(f'Test Set Evaluation for {model} on all datasets based on {metric}')
                plt.plot(raw_stats[metric], label=f'Raw, mean: {raw_stats[metric].mean():.4f}', color='tab:blue')
                plt.plot(morph_stats[metric], label=f'Morph, mean: {morph_stats[metric].mean():.4f}', color='tab:orange')
                plt.plot(otsu_stats[metric], label=f'Otsu, mean: {otsu_stats[metric].mean():.4f}', color='tab:green')

                plt.fill_between(morph_stats['SIC Label'].index, morph_stats['SIC Label'], 0, color='tab:orange', alpha=0.2)
                plt.fill_between(otsu_stats['SIC Label'].index, otsu_stats['SIC Label'], 0, color='tab:green', alpha=0.2)
                plt.fill_between(raw_stats['SIC Label'].index, raw_stats['SIC Label'], 0, color='tab:blue', alpha=0.2)

                plt.xlabel('Image Number')
                plt.ylabel(metric)
                plt.legend()
                plt.grid()
                plt.ylim(0, 1.1)
                plt.savefig(f'{output_path}/{model}_{metric}.png')
                plt.close()

    elif mode=='dataset':
        unet_raw = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet/raw/evaluation_scores.csv')
        unet_morph = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet/morph/evaluation_scores.csv')
        unet_otsu = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet/otsu/evaluation_scores.csv')

        deeplabv3plus_raw = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3plus/raw/evaluation_scores.csv')
        deeplabv3plus_morph = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3plus/morph/evaluation_scores.csv')
        deeplabv3plus_otsu = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3plus/otsu/evaluation_scores.csv')

        segformer_raw = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/segformer/raw/evaluation_scores.csv')
        segformer_morph = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/segformer/morph/evaluation_scores.csv')
        segformer_otsu = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/segformer/otsu/evaluation_scores.csv')

        for metric in metrics:

            fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

            axs[0].plot(unet_raw[metric], label='U-Net', color='tab:blue')
            axs[0].plot(deeplabv3plus_raw[metric], label='DeepLabV3+', color='tab:orange')
            axs[0].plot(segformer_raw[metric], label='SegFormer', color='tab:green')
            axs[0].fill_between(segformer_raw['SIC Label'].index, segformer_raw['SIC Label'], 0, color='gray', alpha=0.2)
            axs[0].set_ylabel(f'Raw {metric}', fontsize=18)
            axs[0].set_ylim(0, 1.1)
            axs[0].legend(fontsize=18)

            axs[1].plot(unet_morph[metric], label=f'Unet Morph, mean: {unet_morph[metric].mean():.4f}', color='tab:blue')
            axs[1].plot(deeplabv3plus_morph[metric], label=f'Deeplabv3plus Morph, mean: {deeplabv3plus_morph[metric].mean():.4f}', color='tab:orange')
            axs[1].plot(segformer_morph[metric], label=f'Segformer Morph, mean: {segformer_morph[metric].mean():.4f}', color='tab:green')
            axs[1].fill_between(segformer_morph['SIC Label'].index, segformer_morph['SIC Label'], 0, color='gray', alpha=0.2)
            axs[1].set_ylim(0, 1.1)
            axs[1].set_ylabel(f'Morph {metric}', fontsize=18)

            axs[2].plot(unet_otsu[metric], label=f'Unet Otsu, mean: {unet_otsu[metric].mean():.4f}', color='tab:blue')
            axs[2].plot(deeplabv3plus_otsu[metric], label=f'Deeplabv3plus Otsu, mean: {deeplabv3plus_otsu[metric].mean():.4f}', color='tab:orange')
            axs[2].plot(segformer_otsu[metric], label=f'Segformer Otsu, mean: {segformer_otsu[metric].mean():.4f}', color='tab:green')
            axs[2].fill_between(segformer_otsu['SIC Label'].index, segformer_otsu['SIC Label'], 0, color='gray', alpha=0.2)
            axs[2].set_ylabel(f'Otsu {metric}', fontsize=18)
            axs[2].set_ylim(0, 1.1)
            axs[2].set_xlabel('Test Set Image #', fontsize=18)

            plt.tight_layout()
            # plt.show()
            plt.savefig(f'{output_path}/all_dataset_{metric}.png')
            plt.close()


if __name__ == '__main__':
    mode = sys.argv[1]
    compare_model_performance(mode=mode)

