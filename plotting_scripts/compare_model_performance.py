
import os 
import sys
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

### options:
# 1. compare each model's performance on the same dataset
# 2. compare a model's performance on each dataset 
# 3. option to view the performance of a model on a specific dataset

input_dict = {'loss': 'BCE Loss',
            'iou': 'IOU',
            'accuracy': 'Pixel Accuracy',
            'dice': 'Dice Coefficient',
            'accuracy': 'Pixel Accuracy',
            'sic': 'SIC Label'
            }


def compare_model_performance(model_name, dataset_name, metric='iou'):


    # 1. compare each model's performance on the same dataset
    if model_name == 'all' and dataset_name != 'all':
        print(f"Comparing all models on dataset: {dataset_name} with metric: {metric}")

        # Find all models
        models = os.listdir('/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output')

        # get all models performance
        unet_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet_brain/{dataset_name}/evaluation_scores.csv')
        deeplabv3_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3/{dataset_name}/evaluation_scores.csv')
        dinov2_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/dinov2/{dataset_name}/evaluation_scores.csv')
        pspnet_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/pspnet/{dataset_name}/evaluation_scores.csv')
        deeplabv3plus_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3plus/{dataset_name}/evaluation_scores.csv')
        unet_smp_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet_smp/{dataset_name}/evaluation_scores.csv')


        plt.title(f"Performance of all models on dataset: {dataset_name} with metric: {metric}")

        plt.plot(unet_eval_scores[input_dict[metric]], label=f'unet {input_dict[metric]}, mean: {unet_eval_scores[input_dict[metric]].mean():.4f}')
        plt.plot(deeplabv3_eval_scores[input_dict[metric]], label=f'deeplabv3 {input_dict[metric]}, mean: {deeplabv3_eval_scores[input_dict[metric]].mean():.4f}')
        plt.plot(dinov2_eval_scores[input_dict[metric]], label=f'dinov2 {input_dict[metric]}, mean: {dinov2_eval_scores[input_dict[metric]].mean():.4f}')
        plt.plot(pspnet_eval_scores[input_dict[metric]], label=f'pspnet {input_dict[metric]}, mean: {pspnet_eval_scores[input_dict[metric]].mean():.4f}')
        plt.plot(deeplabv3plus_eval_scores[input_dict[metric]], label=f'deeplabv3plus {input_dict[metric]}, mean: {deeplabv3plus_eval_scores[input_dict[metric]].mean():.4f}')
        plt.plot(unet_smp_eval_scores[input_dict[metric]], label=f'unet_smp {input_dict[metric]}, mean: {unet_smp_eval_scores[input_dict[metric]].mean():.4f}')


        # plt.plot(unet_eval_scores['SIC Label, %'] / 100, label='SIC, Label')
        
        plt.xlabel('Image')
        plt.ylabel(metric)
        plt.legend()
        plt.grid()

        plt.show()

        pass

    # 2. compare a model's performance on each dataset
    elif model_name != 'all' and dataset_name == 'all':
        print(f"Comparing model: {model_name} on all datasets")

        raw_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model_name}/raw/evaluation_scores.csv')
        morph_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model_name}/morph/evaluation_scores.csv')
        otsu_eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model_name}/otsu/evaluation_scores.csv')

        # lets plot loss and dice

        plt.title(f"{model_name} performance on all datasets")

        plt.plot(raw_eval_scores[input_dict[metric]], label=f'raw {input_dict[metric]}, mean: {raw_eval_scores[input_dict[metric]].mean():.4f}')
        plt.plot(morph_eval_scores[input_dict[metric]], label=f'morph {input_dict[metric]}, mean: {morph_eval_scores[input_dict[metric]].mean():.4f}')
        plt.plot(otsu_eval_scores[input_dict[metric]], label=f'otsu {input_dict[metric]}, mean: {otsu_eval_scores[input_dict[metric]].mean():.4f}')
        
        plt.xlabel('Image')
        plt.ylabel(metric)
        plt.legend()
        plt.grid()

        plt.show()

        pass

    # 3. option to view the performance of a model on a specific dataset
    elif model_name != 'all' and dataset_name != 'all':
        print(f"Comparing model: {model_name} on dataset: {dataset_name}")

        eval_scores = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model_name}/{dataset_name}/evaluation_scores.csv')

        plt.title(f"{model_name} performance on dataset: {dataset_name}")

        plt.plot(eval_scores['BCE Loss'], label='BCE loss')
        plt.plot(eval_scores['IOU'], label='iou')
        plt.plot(eval_scores['Pixel Accuracy'], label='pixel accuracy')
        plt.plot(eval_scores['Dice Coefficient'], label='dice')

        plt.xlabel('Image')
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

    compare_model_performance(model_name, dataset_name, metric)





