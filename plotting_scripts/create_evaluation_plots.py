import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_path = '/home/cole/Pictures/thesis_report/test_set_statistics'

def compare_model_performance():

    models = ['unet', 'deeplabv3plus', 'segformer']
    datasets = ['raw', 'morph', 'otsu']
    metrics = ['IOU', 'DICE', 'Pixel Accuracy', 'Precision', 'Recall']

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

    for dataset in datasets:

        unet_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet/{dataset}/evaluation_scores.csv')
        deeplabv3plus_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3plus/{dataset}/evaluation_scores.csv')
        segformer_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/segformer/{dataset}/evaluation_scores.csv')

        for metric in metrics:

            plt.figure(figsize=(15, 8), facecolor='lightgray')
            # plt.title(f'Test Set Evaluation for all models on {dataset} based on {metric}')
            plt.plot(unet_stats[metric], label=f'Unet, mean: {unet_stats[metric].mean():.4f}', color='tab:blue')
            plt.plot(deeplabv3plus_stats[metric], label=f'Deeplabv3plus, mean: {deeplabv3plus_stats[metric].mean():.4f}', color='tab:orange')
            plt.plot(segformer_stats[metric], label=f'Segformer, mean: {segformer_stats[metric].mean():.4f}', color='tab:green')

            plt.fill_between(deeplabv3plus_stats['SIC Label'].index, deeplabv3plus_stats['SIC Label'], 0, color='gray', alpha=0.2)

            plt.xlabel('Image Number', fontsize=18)
            plt.ylabel(metric, fontsize=18)
            plt.legend(fontsize=18)
            plt.grid()
            plt.tight_layout()
            plt.ylim(0, 1.1)
            plt.savefig(f'{output_path}/{dataset}_{metric}.png')
            plt.close()


if __name__ == '__main__':
    compare_model_performance()

