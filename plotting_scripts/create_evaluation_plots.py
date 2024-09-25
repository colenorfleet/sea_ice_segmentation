import os 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Goals:
- create confusion matrix with avg. values for each model/dataset (maybe just a bar graph for now)
- for each model, one plot of IOU and F1 score for each dataset
- and one plot of IOU and F1 score for each model
- and same thing for foreground vs background accuracy
'''


def compare_model_performance():

    # get all models performance
    unet_eval_scores_raw = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet/raw/evaluation_scores.csv')
    unet_eval_scores_morph = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet/morph/evaluation_scores.csv')
    unet_eval_scores_otsu = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/unet/otsu/evaluation_scores.csv')

    deeplabv3plus_eval_scores_raw = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3plus/raw/evaluation_scores.csv')
    deeplabv3plus_eval_scores_morph = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3plus/morph/evaluation_scores.csv')
    deeplabv3plus_eval_scores_otsu = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/deeplabv3plus/otsu/evaluation_scores.csv')

    segformer_eval_scores_raw = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/segformer/raw/evaluation_scores.csv')
    segformer_eval_scores_morph = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/segformer/morph/evaluation_scores.csv')
    segformer_eval_scores_otsu = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/segformer/otsu/evaluation_scores.csv')


    # plot IOU 
    plt.figure(figsize=(20, 10))
    plt.title('IOU, Raw')
    plt.plot(unet_eval_scores_raw['IOU'], label='unet IOU, mean: {:.4f}'.format(unet_eval_scores_raw['IOU'].mean()))
    plt.plot(deeplabv3plus_eval_scores_raw['IOU'], label='deeplabv3plus IOU, mean: {:.4f}'.format(deeplabv3plus_eval_scores_raw['IOU'].mean()))
    plt.plot(segformer_eval_scores_raw['IOU'], label='segformer IOU, mean: {:.4f}'.format(segformer_eval_scores_raw['IOU'].mean()))
    plt.xlabel('Image')
    plt.ylabel('IOU')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.1)
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/iou/IOU_raw.png')

    plt.figure(figsize=(20, 10))
    plt.title('IOU, Morph')
    plt.plot(unet_eval_scores_morph['IOU'], label='unet IOU, mean: {:.4f}'.format(unet_eval_scores_morph['IOU'].mean()))
    plt.plot(deeplabv3plus_eval_scores_morph['IOU'], label='deeplabv3plus IOU, mean: {:.4f}'.format(deeplabv3plus_eval_scores_morph['IOU'].mean()))
    plt.plot(segformer_eval_scores_morph['IOU'], label='segformer IOU, mean: {:.4f}'.format(segformer_eval_scores_morph['IOU'].mean()))
    plt.xlabel('Image')
    plt.ylabel('IOU')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.1)
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/iou/IOU_morph.png')

    plt.figure(figsize=(20, 10))
    plt.title('IOU, Otsu')
    plt.plot(unet_eval_scores_otsu['IOU'], label='unet IOU, mean: {:.4f}'.format(unet_eval_scores_otsu['IOU'].mean()))
    plt.plot(deeplabv3plus_eval_scores_otsu['IOU'], label='deeplabv3plus IOU, mean: {:.4f}'.format(deeplabv3plus_eval_scores_otsu['IOU'].mean()))
    plt.plot(segformer_eval_scores_otsu['IOU'], label='segformer IOU, mean: {:.4f}'.format(segformer_eval_scores_otsu['IOU'].mean()))
    plt.xlabel('Image')
    plt.ylabel('IOU')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.1)
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/iou/IOU_otsu.png')

    # plot F1 score
    plt.figure(figsize=(20, 10))
    plt.title('F1 Score, Raw')
    plt.plot(unet_eval_scores_raw['F1 Score'], label='unet F1 Score, mean: {:.4f}'.format(unet_eval_scores_raw['F1 Score'].mean()))
    plt.plot(deeplabv3plus_eval_scores_raw['F1 Score'], label='deeplabv3plus F1 Score, mean: {:.4f}'.format(deeplabv3plus_eval_scores_raw['F1 Score'].mean()))
    plt.plot(segformer_eval_scores_raw['F1 Score'], label='segformer F1 Score, mean: {:.4f}'.format(segformer_eval_scores_raw['F1 Score'].mean()))
    plt.xlabel('Image')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.1)
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/f1/F1_raw.png')

    plt.figure(figsize=(20, 10))
    plt.title('F1 Score, Morph')
    plt.plot(unet_eval_scores_morph['F1 Score'], label='unet F1 Score, mean: {:.4f}'.format(unet_eval_scores_morph['F1 Score'].mean()))
    plt.plot(deeplabv3plus_eval_scores_morph['F1 Score'], label='deeplabv3plus F1 Score, mean: {:.4f}'.format(deeplabv3plus_eval_scores_morph['F1 Score'].mean()))
    plt.plot(segformer_eval_scores_morph['F1 Score'], label='segformer F1 Score, mean: {:.4f}'.format(segformer_eval_scores_morph['F1 Score'].mean()))
    plt.xlabel('Image')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.1)
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/f1/F1_morph.png')

    plt.figure(figsize=(20, 10))
    plt.title('F1 Score, Otsu')
    plt.plot(unet_eval_scores_otsu['F1 Score'], label='unet F1 Score, mean: {:.4f}'.format(unet_eval_scores_otsu['F1 Score'].mean()))
    plt.plot(deeplabv3plus_eval_scores_otsu['F1 Score'], label='deeplabv3plus F1 Score, mean: {:.4f}'.format(deeplabv3plus_eval_scores_otsu['F1 Score'].mean()))
    plt.plot(segformer_eval_scores_otsu['F1 Score'], label='segformer F1 Score, mean: {:.4f}'.format(segformer_eval_scores_otsu['F1 Score'].mean()))
    plt.xlabel('Image')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.1)
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/f1/F1_otsu.png')

    # plot foreground accuracy
    plt.figure(figsize=(20, 10))
    plt.title('Foreground Accuracy, Raw')
    plt.plot(unet_eval_scores_raw['Foreground Accuracy'], label='unet Foreground Accuracy, mean: {:.4f}'.format(unet_eval_scores_raw['Foreground Accuracy'].mean()))
    plt.plot(deeplabv3plus_eval_scores_raw['Foreground Accuracy'], label='deeplabv3plus Foreground Accuracy, mean: {:.4f}'.format(deeplabv3plus_eval_scores_raw['Foreground Accuracy'].mean()))
    plt.plot(segformer_eval_scores_raw['Foreground Accuracy'], label='segformer Foreground Accuracy, mean: {:.4f}'.format(segformer_eval_scores_raw['Foreground Accuracy'].mean()))
    plt.ylim(0, 1.1)
    plt.xlabel('Image')
    plt.ylabel('Foreground Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/fg_acc/foreground_accuracy_raw.png')

    plt.figure(figsize=(20, 10))
    plt.title('Foreground Accuracy, Morph')
    plt.plot(unet_eval_scores_morph['Foreground Accuracy'], label='unet Foreground Accuracy, mean: {:.4f}'.format(unet_eval_scores_morph['Foreground Accuracy'].mean()))
    plt.plot(deeplabv3plus_eval_scores_morph['Foreground Accuracy'], label='deeplabv3plus Foreground Accuracy, mean: {:.4f}'.format(deeplabv3plus_eval_scores_morph['Foreground Accuracy'].mean()))
    plt.plot(segformer_eval_scores_morph['Foreground Accuracy'], label='segformer Foreground Accuracy, mean: {:.4f}'.format(segformer_eval_scores_morph['Foreground Accuracy'].mean()))
    plt.ylim(0, 1.1)
    plt.xlabel('Image')
    plt.ylabel('Foreground Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/fg_acc/foreground_accuracy_morph.png')

    plt.figure(figsize=(20, 10))
    plt.title('Foreground Accuracy, Otsu')
    plt.plot(unet_eval_scores_otsu['Foreground Accuracy'], label='unet Foreground Accuracy, mean: {:.4f}'.format(unet_eval_scores_otsu['Foreground Accuracy'].mean()))
    plt.plot(deeplabv3plus_eval_scores_otsu['Foreground Accuracy'], label='deeplabv3plus Foreground Accuracy, mean: {:.4f}'.format(deeplabv3plus_eval_scores_otsu['Foreground Accuracy'].mean()))
    plt.plot(segformer_eval_scores_otsu['Foreground Accuracy'], label='segformer Foreground Accuracy, mean: {:.4f}'.format(segformer_eval_scores_otsu['Foreground Accuracy'].mean()))
    plt.ylim(0, 1.1)
    plt.xlabel('Image')
    plt.ylabel('Foreground Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('/home/cole/Documents/NTNU/sea_ice_segmentation/metric_plots/evaluating/fg_acc/foreground_accuracy_otsu.png')

    



if __name__ == '__main__':
    compare_model_performance()

