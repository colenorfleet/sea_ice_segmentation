import os 
import sys
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

input_dict = {'Loss': ['Avg BCE Train Loss', 'Avg BCE Val Loss'],
              'IOU': ['Avg Train IOU', 'Avg Val IOU'],
                'DICE': ['Avg Train DICE', 'Avg Val DICE'],
}

model_names = ['unet', 'deeplabv3plus', 'segformer']
dataset_names = ['raw', 'morph', 'otsu']

output_dir = '15_epoch_oct_10'

def view_training_statistics(model_name, dataset_name, metric='loss', save=False):

    # view stats for each model on a specific dataset

    if model_name != 'all' and dataset_name == 'all':

        # get the training statistics for the model

        raw_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/raw/training_logs.csv')
        morph_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/morph/training_logs.csv')
        otsu_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/otsu/training_logs.csv')

        number_epochs = len(raw_train_stats['Avg BCE Train Loss'])-1

        plt.figure(figsize=(10, 5), facecolor='lightgray')

        # plt.title(f"{metric} training statistics for {model_name} on ALL datasets", size=20)

        plt.plot(raw_train_stats[input_dict[metric][0]], label=f'raw dataset training {metric}', color='tab:blue')
        plt.plot(morph_train_stats[input_dict[metric][0]], label=f'morph dataset training {metric}', color='tab:orange')
        plt.plot(otsu_train_stats[input_dict[metric][0]], label=f'otsu dataset training {metric}', color='tab:green')

        plt.plot(raw_train_stats[input_dict[metric][1]], label=f'raw dataset validation {metric}', color = 'tab:blue', linestyle='--')
        plt.plot(morph_train_stats[input_dict[metric][1]], label=f'morph dataset validation {metric}', color='tab:orange', linestyle='--')
        plt.plot(otsu_train_stats[input_dict[metric][1]], label=f'otsu dataset validation {metric}', color='tab:green', linestyle='--')


        plt.xlabel('Epoch', size=10)
        plt.ylabel(f"{metric}", size=10)

        plt.xlim(0, number_epochs)
        plt.ylim(0, 1.25)

        plt.legend(fontsize=10)
        plt.grid()

        if save == 'True':
            plt.savefig(f'/home/cole/Pictures/thesis_report/training_statistics/{output_dir}/{model_name}_training_{metric}.png')
            plt.close()
        else:
            plt.show()

        pass

    elif model_name == 'all' and dataset_name in dataset_names:

        # get the training statistics for each model

        unet_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/unet/{dataset_name}/training_logs.csv')
        deeplabv3plus_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/deeplabv3plus/{dataset_name}/training_logs.csv')
        segformer_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/segformer/{dataset_name}/training_logs.csv')

        number_epochs = len(unet_train_stats['Avg BCE Train Loss'])-1
        # For now let's just plot loss

        plt.figure(figsize=(6, 4), facecolor='lightgray')

        # plt.title(f"{metric} training statistics for all models on dataset: {dataset_name}", size=20)

        plt.plot(unet_train_stats[input_dict[metric][0]], label=f'unet training {metric}', color='tab:blue')
        plt.plot(unet_train_stats[input_dict[metric][1]], label=f'unet validation {metric}', color='tab:blue', linestyle='--')

        plt.plot(deeplabv3plus_train_stats[input_dict[metric][0]], label=f'deeplabv3plus training {metric}', color='tab:orange')
        plt.plot(deeplabv3plus_train_stats[input_dict[metric][1]], label=f'deeplabv3plus validation {metric}', color='tab:orange', linestyle='--')

        plt.plot(segformer_train_stats[input_dict[metric][0]], label=f'segformer training {metric}', color='tab:green')
        plt.plot(segformer_train_stats[input_dict[metric][1]], label=f'segformer validation {metric}', color='tab:green', linestyle='--')

        plt.xlabel('Epoch', size=10)
        plt.ylabel(f'{metric}', size=10)

        plt.xlim(0, number_epochs)
        plt.ylim(0, 1.25)

        plt.legend(fontsize=10)
        plt.grid()

        if save == 'True':
            plt.savefig(f'/home/cole/Pictures/thesis_report/training_statistics/{output_dir}/{dataset_name}_training_{metric}.png')
            plt.close()
        else:
            plt.show()

        pass


    # view stats for a specific model on a specific dataset

    elif model_name != 'all' and dataset_name in dataset_names:

        # get the training statistics for the model

        train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/output/{model_name}/{dataset_name}/training_logs.csv')

        plt.title(f"Training statistics for model: {model_name} on dataset: {dataset_name}")

        plt.plot(train_stats[input_dict[metric][0]], label=f'{model_name} training {metric}')
        plt.plot(train_stats[input_dict[metric[1]]], label=f'{model_name} validation {metric}')

        plt.xlabel('Epoch')
        plt.ylabel(f"{metric}")

        plt.legend()
        plt.grid()

        plt.show()

        pass

    elif model_name == 'all' and dataset_name == 'trained_on_all':

        # get the training statistics for each model

        unet_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/all_dataset_output/unet/training_logs.csv')
        deeplabv3plus_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/all_dataset_output/deeplabv3plus/training_logs.csv')
        segformer_train_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/all_dataset_output/segformer/training_logs.csv')

        number_epochs = len(unet_train_stats['Epoch'])
        # For now let's just plot loss

        plt.figure(figsize=(20, 10))

        plt.title(f"{metric} training statistics for all models trained all datasets in sequence")

        plt.vlines([4, 9], 0, 1.25, color='black', linestyle='--')
        

        plt.plot(unet_train_stats[input_dict[metric][0]], label=f'unet training {metric}', color='tab:blue')
        plt.plot(unet_train_stats[input_dict[metric][1]], label=f'unet validation {metric}', color='tab:blue', linestyle='--')

        plt.plot(deeplabv3plus_train_stats[input_dict[metric][0]], label=f'deeplabv3plus training {metric}', color='tab:orange')
        plt.plot(deeplabv3plus_train_stats[input_dict[metric][1]], label=f'deeplabv3plus validation {metric}', color='tab:orange', linestyle='--')

        plt.plot(segformer_train_stats[input_dict[metric][0]], label=f'segformer training {metric}', color='tab:green')
        plt.plot(segformer_train_stats[input_dict[metric][1]], label=f'segformer validation {metric}', color='tab:green', linestyle='--')

        plt.text(1.5, -0.1, 'Raw Dataset', fontsize=12)
        plt.text(5.5, -0.1, 'Morph Dataset', fontsize=12)
        plt.text(10.5, -0.1, 'Otsu Dataset', fontsize=12)

        plt.xlabel('Epoch')
        plt.ylabel(f'{metric}')

        plt.xlim(0, number_epochs)
        # plt.ylim(0, 1.25)

        plt.legend()
        plt.grid()
        
        if save == 'True':
            plt.savefig(f'/home/cole/Pictures/thesis_report/training_statistics/all_dataset_training_{metric}.png')
            plt.close()
        else:
            plt.show()

        


    else:
        print("Invalid input")
        return
    

if __name__ == "__main__":

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    metric = sys.argv[3]
    save_flag = sys.argv[4]

    view_training_statistics(model_name, dataset_name, metric, save_flag)