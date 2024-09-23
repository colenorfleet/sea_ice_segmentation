import os 
import sys
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

input_dict = {'train': 'Avg BCE Train Loss',
              'val': 'Avg BCE Val Loss'}

def view_training_statistics(model_name, dataset_name, metric='train'):

    # view stats for each model on a specific dataset

    if model_name != 'all' and dataset_name == 'all' and metric == 'both':

        # get the training statistics for the model

        raw_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/raw/training_logs.csv')
        morph_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/morph/training_logs.csv')
        otsu_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/otsu/training_logs.csv')

        plt.figure(figsize=(20, 10))

        plt.title(f"Losses for model: {model_name} on all datasets")

        plt.plot(raw_train_stats['Avg BCE Train Loss'], label=f'raw dataset train loss', color='red')
        plt.plot(morph_train_stats['Avg BCE Train Loss'], label=f'morph train loss', color='blue')
        plt.plot(otsu_train_stats['Avg BCE Train Loss'], label=f'otsu train loss', color='green')

        plt.plot(raw_train_stats['Avg BCE Val Loss'], label=f'raw dataset val loss', color = 'red', linestyle='--')
        plt.plot(morph_train_stats['Avg BCE Val Loss'], label=f'morph val loss', color='blue', linestyle='--')
        plt.plot(otsu_train_stats['Avg BCE Val Loss'], label=f'otsu val loss', color='green', linestyle='--')

        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')

        plt.legend()
        plt.grid()

        plt.savefig(f'/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/training/{model_name}_losses.png')

        plt.show()

        pass

    elif model_name == 'all' and dataset_name != 'all' and metric == 'both':

        # get the training statistics for each model

        unet_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/unet_brain/{dataset_name}/training_logs.csv')
        deeplabv3_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/deeplabv3/{dataset_name}/training_logs.csv')
        dinov2_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/dinov2/{dataset_name}/training_logs.csv')
        pspnet_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/pspnet/{dataset_name}/training_logs.csv')
        deeplabv3plus_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/deeplabv3plus/{dataset_name}/training_logs.csv')
        unet_smp_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/unet_smp/{dataset_name}/training_logs.csv')


        # For now let's just plot loss

        plt.figure(figsize=(20, 10))


        plt.title(f"Losses for all models on dataset: {dataset_name}")

        plt.plot(unet_train_stats[input_dict['train']], label=f'unet train loss', color='red')
        plt.plot(unet_train_stats[input_dict['val']], label=f'unet val loss', color='red', linestyle='--')

        plt.plot(deeplabv3_train_stats[input_dict['train']], label=f'deeplabv3 train loss', color='cyan')
        plt.plot(deeplabv3_train_stats[input_dict['val']], label=f'deeplabv3 val loss', color='cyan', linestyle='--')

        plt.plot(dinov2_train_stats[input_dict['train']], label=f'dinov2 train loss', color='gold')
        plt.plot(dinov2_train_stats[input_dict['val']], label=f'dinov2 val loss', color='gold', linestyle='--')

        plt.plot(pspnet_train_stats[input_dict['train']], label=f'pspnet train loss', color='green')
        plt.plot(pspnet_train_stats[input_dict['val']], label=f'pspnet val loss', color='green', linestyle='--')

        plt.plot(deeplabv3plus_train_stats[input_dict['train']], label=f'deeplabv3plus train loss', color='blue')
        plt.plot(deeplabv3plus_train_stats[input_dict['val']], label=f'deeplabv3plus val loss', color='blue', linestyle='--')

        plt.plot(unet_smp_train_stats[input_dict['train']], label=f'unet_smp train loss', color='purple')
        plt.plot(unet_smp_train_stats[input_dict['val']], label=f'unet_smp val loss', color='purple', linestyle='--')

        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')

        plt.legend()
        plt.grid()

        plt.savefig(f'/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/training/{dataset_name}_losses.png')

        plt.show()

        pass

    # view stats for a specific model on all datasets
    elif model_name != 'all' and dataset_name == 'all':

        # get the training statistics for the model

        raw_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/raw/training_logs.csv')
        morph_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/morph/training_logs.csv')
        otsu_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/otsu/training_logs.csv')

        plt.title(f"{metric} loss for model: {model_name} on all datasets")

        plt.plot(raw_train_stats[input_dict[metric]], label=f'raw dataset {metric} loss')
        plt.plot(morph_train_stats[input_dict[metric]], label=f'morph {metric} loss')
        plt.plot(otsu_train_stats[input_dict[metric]], label=f'otsu {metric} loss')

        plt.xlabel('Epoch')
        plt.ylabel(f'BCE {metric} Loss')

        plt.legend()
        plt.grid()

        plt.show()

        pass


    # view stats for a specific model on a specific dataset

    elif model_name != 'all' and dataset_name != 'all':

        # get the training statistics for the model

        train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/{dataset_name}/training_logs.csv')

        plt.title(f"Training statistics for model: {model_name} on dataset: {dataset_name}")

        plt.plot(train_stats['Avg BCE Train Loss'], label=f'{model_name} train loss')
        plt.plot(train_stats['Avg BCE Val Loss'], label=f'{model_name} val loss')

        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')

        plt.legend()
        plt.grid()

        plt.show()

        pass


    




    else:
        print("Invalid input")
        return
    

if __name__ == "__main__":

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    metric = sys.argv[3]

    view_training_statistics(model_name, dataset_name, metric)