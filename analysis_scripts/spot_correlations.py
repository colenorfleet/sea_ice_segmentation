import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools
import seaborn as sns


def spot_correlations(model):

    metric_dict = {
        'loss': 'BCE Loss',
        'total_loss': 'Total BCE Loss',
        'iou': 'IOU',
        'dice': 'DICE',
        'pix_acc': 'Pixel Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'numTP': 'Number True Positive',
        'numFP': 'Number False Positive',
        'numTN': 'Number True Negative',
        'numFN': 'Number False Negative',
        'sic_label': 'SIC Label',
        'sic_pred': 'SIC Pred'
    }


    goNorth_raw = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/512x512/labelled_output/{model}/goNorth/raw/evaluation_scores.csv')
    goNorth_morph = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/512x512/labelled_output/{model}/goNorth/morph/evaluation_scores.csv')
    goNorth_otsu = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/512x512/labelled_output/{model}/goNorth/otsu/evaluation_scores.csv')
    roboflow_raw = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/512x512/labelled_output/{model}/roboflow/raw/evaluation_scores.csv')
    roboflow_morph = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/512x512/labelled_output/{model}/roboflow/morph/evaluation_scores.csv')
    roboflow_otsu = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/512x512/labelled_output/{model}/roboflow/otsu/evaluation_scores.csv')

    raw = pd.concat([goNorth_raw, roboflow_raw])
    morph = pd.concat([goNorth_morph, roboflow_morph])
    otsu = pd.concat([goNorth_otsu, roboflow_otsu])

    # Get correlation matrix
    raw_corr_matrix = raw.corr().abs()
    morph_corr_matrix = morph.corr().abs()
    otsu_corr_matrix = otsu.corr().abs()

    # Select upper triangle of correlation matrix
    raw_upper_triangle = raw_corr_matrix.where(np.triu(np.ones(raw_corr_matrix.shape), k=1).astype(bool))
    morph_upper_triangle = morph_corr_matrix.where(np.triu(np.ones(morph_corr_matrix.shape), k=1).astype(bool))
    otsu_upper_triangle = otsu_corr_matrix.where(np.triu(np.ones(otsu_corr_matrix.shape), k=1).astype(bool))

    # Stack the upper triangle to get pairs
    raw_corr_pairs = raw_upper_triangle.stack()
    morph_corr_pairs = morph_upper_triangle.stack()
    otsu_corr_pairs = otsu_upper_triangle.stack()

    # Convert to dictionary
    raw_corr_dict = { (col1, col2): corr_value for (col1, col2), corr_value in raw_corr_pairs.items() }
    morph_corr_dict = { (col1, col2): corr_value for (col1, col2), corr_value in morph_corr_pairs.items() }
    otsu_corr_dict = { (col1, col2): corr_value for (col1, col2), corr_value in otsu_corr_pairs.items() }

    raw_sorted_corr = sorted(raw_corr_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    morph_sorted_corr = sorted(morph_corr_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    otsu_sorted_corr = sorted(otsu_corr_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    for i in range(5):
        print(f'RAW: {raw_sorted_corr[i]}')
        print(f'MORPH: {morph_sorted_corr[i]}')
        print(f'OTSU: {otsu_sorted_corr[i]}')


    fig, axs = plt.subplots(1, 3, figsize=(30, 10), sharey=True)
    sns.heatmap(raw.corr(), annot=True, cmap='coolwarm', cbar=False, ax=axs[0])
    sns.heatmap(morph.corr(), annot=True, cmap='coolwarm', cbar=False, ax=axs[1])
    sns.heatmap(otsu.corr(), annot=True, cmap='coolwarm', cbar=False, ax=axs[2])
    plt.show()
    exit()


    '''
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    axs[0].scatter(raw[metric_dict[metric1]], raw[metric_dict[metric2]], s=10)
    axs[0].plot(raw[metric_dict[metric1]], raw_fit.coef_*raw[metric_dict[metric1]] + raw_fit.intercept_, color='red')
    axs[0].set_title(f'Raw, {round(raw_rq, 2)}')
    axs[0].set_ylabel(metric_dict[metric2])
    axs[0].set_ylim([0, 1.1])

    axs[1].scatter(morph[metric_dict[metric1]], morph[metric_dict[metric2]], s=10)
    axs[1].plot(morph[metric_dict[metric1]], morph_fit.coef_*morph[metric_dict[metric1]] + morph_fit.intercept_, color='red')
    axs[1].set_title(f'Morph, {round(morph_rq, 2)}')
    axs[1].set_xlabel(metric_dict[metric1])
    axs[1].set_ylim([0, 1.1])


    axs[2].scatter(otsu[metric_dict[metric1]], otsu[metric_dict[metric2]], s=10)
    axs[2].plot(otsu[metric_dict[metric1]], otsu_fit.coef_*otsu[metric_dict[metric1]] + otsu_fit.intercept_, color='red')
    axs[2].set_title(f'Otsu, {round(otsu_rq, 2)}')
    axs[2].set_ylim([0, 1.1])


    plt.tight_layout()
    plt.show()
    plt.close()
    '''



if __name__ == '__main__':

    model = sys.argv[1]

    spot_correlations(model)



    

