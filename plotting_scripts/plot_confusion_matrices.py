import os 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





def plot_condusion_matrices():

    models = {'unet_brain', 'deeplabv3', 'dinov2', 'pspnet', 'deeplabv3plus', 'unet_smp'}

    for model in models:

        raw_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/raw/evaluation_scores.csv')
        morph_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/morph/evaluation_scores.csv')
        otsu_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/otsu/evaluation_scores.csv')

        datasets = {'raw', 'morph', 'otsu'}
        categories = {'TP', 'FP', 'TN', 'FN'}

        # order: TP, FP, TN, FN
        averages = {
            'raw': (int(raw_stats['Number True Positive'].mean()), 
                    int(raw_stats['Number False Positive'].mean()), 
                    int(raw_stats['Number True Negative'].mean()),
                    int(raw_stats['Number False Negative'].mean())),
            'morph': (int(morph_stats['Number True Positive'].mean()),
                      int(morph_stats['Number False Positive'].mean()),
                      int(morph_stats['Number True Negative'].mean()),
                      int(morph_stats['Number False Negative'].mean())),
            'otsu': (int(otsu_stats['Number True Positive'].mean()),
                     int(otsu_stats['Number False Positive'].mean()),
                     int(otsu_stats['Number True Negative'].mean()),
                     int(otsu_stats['Number False Negative'].mean())),
        }
           
        x = np.arange(len(categories))
        width = 0.15
        
        fig, ax = plt.subplots(layout='constrained', figsize=(20, 10))
        multiplier = 0

        for attribute, measurement in averages.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_xlabel('Pixel Classification')
        ax.set_ylabel('Number of Pixels')
        ax.set_title(f'Average Confusion Matrix Values for {model}')
        ax.set_xticks(x + width, ['True Positive', 'False Positive', 'True Negative', 'False Negative'])
        ax.legend(loc='upper left', ncols=4)
        ax.set_ylim(0, 200000)


        plt.savefig(f'/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/confusion_matrices/{model}_confusion_matrix.png')
        plt.close()

if __name__ == '__main__':
    plot_condusion_matrices()