import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

# seems to be some correlations between:
# sic_label, iou, loss

def spot_correlations(model, metric1, metric2, mode):

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

    unet_goNorth_raw = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/unet/goNorth/raw/evaluation_scores.csv')
    unet_goNorth_morph = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/unet/goNorth/morph/evaluation_scores.csv')
    unet_goNorth_otsu = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/unet/goNorth/otsu/evaluation_scores.csv')
    unet_roboflow_raw = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/unet/roboflow/raw/evaluation_scores.csv')
    unet_roboflow_morph = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/unet/roboflow/morph/evaluation_scores.csv')
    unet_roboflow_otsu = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/unet/roboflow/otsu/evaluation_scores.csv')

    deeplabv3plus_goNorth_raw = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/deeplabv3plus/goNorth/raw/evaluation_scores.csv')
    deeplabv3plus_goNorth_morph = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/deeplabv3plus/goNorth/morph/evaluation_scores.csv')
    deeplabv3plus_goNorth_otsu = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/deeplabv3plus/goNorth/otsu/evaluation_scores.csv')
    deeplabv3plus_roboflow_raw = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/deeplabv3plus/roboflow/raw/evaluation_scores.csv')
    deeplabv3plus_roboflow_morph = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/deeplabv3plus/roboflow/morph/evaluation_scores.csv')
    deeplabv3plus_roboflow_otsu = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/deeplabv3plus/roboflow/otsu/evaluation_scores.csv')

    segformer_goNorth_raw = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/segformer/goNorth/raw/evaluation_scores.csv')
    segformer_goNorth_morph = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/segformer/goNorth/morph/evaluation_scores.csv')
    segformer_goNorth_otsu = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/segformer/goNorth/otsu/evaluation_scores.csv')
    segformer_roboflow_raw = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/segformer/roboflow/raw/evaluation_scores.csv')
    segformer_roboflow_morph = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/segformer/roboflow/morph/evaluation_scores.csv')
    segformer_roboflow_otsu = pd.read_csv('/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/segformer/roboflow/otsu/evaluation_scores.csv')

    unet_raw = pd.concat([unet_goNorth_raw, unet_roboflow_raw])
    unet_morph = pd.concat([unet_goNorth_morph, unet_roboflow_morph])
    unet_otsu = pd.concat([unet_goNorth_otsu, unet_roboflow_otsu])

    deeplabv3plus_raw = pd.concat([deeplabv3plus_goNorth_raw, deeplabv3plus_roboflow_raw])
    deeplabv3plus_morph = pd.concat([deeplabv3plus_goNorth_morph, deeplabv3plus_roboflow_morph])
    deeplabv3plus_otsu = pd.concat([deeplabv3plus_goNorth_otsu, deeplabv3plus_roboflow_otsu])

    segformer_raw = pd.concat([segformer_goNorth_raw, segformer_roboflow_raw])
    segformer_morph = pd.concat([segformer_goNorth_morph, segformer_roboflow_morph])
    segformer_otsu = pd.concat([segformer_goNorth_otsu, segformer_roboflow_otsu])

    # Get correlation matrix
    unet_raw_corr_matrix = unet_raw.corr().abs()
    unet_morph_corr_matrix = unet_morph.corr().abs()
    unet_otsu_corr_matrix = unet_otsu.corr().abs()

    deeplabv3plus_raw_corr_matrix = deeplabv3plus_raw.corr().abs()
    deeplabv3plus_morph_corr_matrix = deeplabv3plus_morph.corr().abs()
    deeplabv3plus_otsu_corr_matrix = deeplabv3plus_otsu.corr().abs()

    segformer_raw_corr_matrix = segformer_raw.corr().abs()
    segformer_morph_corr_matrix = segformer_morph.corr().abs()
    segformer_otsu_corr_matrix = segformer_otsu.corr().abs()

    unet_raw_corr_sic = unet_raw_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)
    unet_morph_corr_sic = unet_morph_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)
    unet_otsu_corr_sic = unet_otsu_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)

    deeplabv3plus_raw_corr_sic = deeplabv3plus_raw_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)
    deeplabv3plus_morph_corr_sic = deeplabv3plus_morph_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)
    deeplabv3plus_otsu_corr_sic = deeplabv3plus_otsu_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)

    segformer_raw_corr_sic = segformer_raw_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)
    segformer_morph_corr_sic = segformer_morph_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)
    segformer_otsu_corr_sic = segformer_otsu_corr_matrix['SIC Label'].drop('SIC Label').abs().sort_values(ascending=False)

    '''
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
    '''

    if mode == 'bar':
        fig, axs = plt.subplots(3, 3, figsize=(30, 10), sharey=True)

        unet_raw_corr_sic.plot(kind='bar', ax=axs[0, 0])
        unet_morph_corr_sic.plot(kind='bar', ax=axs[0, 1])
        unet_otsu_corr_sic.plot(kind='bar', ax=axs[0, 2])

        deeplabv3plus_raw_corr_sic.plot(kind='bar', ax=axs[1, 0])
        deeplabv3plus_morph_corr_sic.plot(kind='bar', ax=axs[1, 1])
        deeplabv3plus_otsu_corr_sic.plot(kind='bar', ax=axs[1, 2])

        segformer_raw_corr_sic.plot(kind='bar', ax=axs[2, 0])
        segformer_morph_corr_sic.plot(kind='bar', ax=axs[2, 1])
        segformer_otsu_corr_sic.plot(kind='bar', ax=axs[2, 2])

        plt.tight_layout()
        plt.show()
        exit()
    elif mode == 'scatter':
        
        fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)

        axs[0, 0].scatter(unet_roboflow_raw[metric_dict[metric1]], unet_roboflow_raw[metric_dict[metric2]], s=10, color='tab:blue')
        axs[0, 0].scatter(unet_goNorth_raw[metric_dict[metric1]], unet_goNorth_raw[metric_dict[metric2]], s=10, color='tab:red')
        axs[0, 0].set_title('Raw')
        axs[0, 0].set_ylabel(f'U-Net, {metric_dict[metric2]}')
        axs[0, 0].set_xlim([0, 1])
        axs[0, 0].set_facecolor('0.9')
        # axs[0].set_ylim([0, 1.1])
        axs[0, 1].scatter(unet_roboflow_morph[metric_dict[metric1]], unet_roboflow_morph[metric_dict[metric2]], s=10, color='tab:orange')
        axs[0, 1].scatter(unet_goNorth_morph[metric_dict[metric1]], unet_goNorth_morph[metric_dict[metric2]], s=10, color='tab:purple')
        axs[0, 1].set_title('Morph')
        axs[0, 1].set_xlim([0, 1])
        axs[0, 1].set_facecolor('0.9')
        # axs[1].set_ylim([0, 1.1])
        axs[0, 2].scatter(unet_roboflow_otsu[metric_dict[metric1]], unet_roboflow_otsu[metric_dict[metric2]], s=10, color='tab:green')
        axs[0, 2].scatter(unet_goNorth_otsu[metric_dict[metric1]], unet_goNorth_otsu[metric_dict[metric2]], s=10, color='tab:brown')
        axs[0, 2].set_title('Otsu')
        axs[0, 2].set_xlim([0, 1])
        axs[0, 2].set_facecolor('0.9')
        # axs[2].set_ylim([0, 1.1])

        axs[1, 0].scatter(deeplabv3plus_roboflow_raw[metric_dict[metric1]], deeplabv3plus_roboflow_raw[metric_dict[metric2]], s=10, color='tab:blue')
        axs[1, 0].scatter(deeplabv3plus_goNorth_raw[metric_dict[metric1]], deeplabv3plus_goNorth_raw[metric_dict[metric2]], s=10, color='tab:red')
        axs[1, 0].set_ylabel(f'DeepLabV3+, {metric_dict[metric2]}')
        axs[1, 0].set_xlim([0, 1])
        axs[1, 0].set_facecolor('0.9')
        axs[1, 1].scatter(deeplabv3plus_roboflow_morph[metric_dict[metric1]], deeplabv3plus_roboflow_morph[metric_dict[metric2]], s=10, color='tab:orange')
        axs[1, 1].scatter(deeplabv3plus_goNorth_morph[metric_dict[metric1]], deeplabv3plus_goNorth_morph[metric_dict[metric2]], s=10, color='tab:purple')
        axs[1, 1].set_xlim([0, 1])
        axs[1, 1].set_facecolor('0.9')
        axs[1, 2].scatter(deeplabv3plus_roboflow_otsu[metric_dict[metric1]], deeplabv3plus_roboflow_otsu[metric_dict[metric2]], s=10, color='tab:green')
        axs[1, 2].scatter(deeplabv3plus_goNorth_otsu[metric_dict[metric1]], deeplabv3plus_goNorth_otsu[metric_dict[metric2]], s=10, color='tab:brown')
        axs[1, 2].set_facecolor('0.9')
        axs[1, 2].set_xlim([0, 1])

        axs[2, 0].scatter(segformer_roboflow_raw[metric_dict[metric1]], segformer_roboflow_raw[metric_dict[metric2]], s=10, color='tab:blue')
        axs[2, 0].scatter(segformer_goNorth_raw[metric_dict[metric1]], segformer_goNorth_raw[metric_dict[metric2]], s=10, color='tab:red')
        axs[2, 0].set_ylabel(f'SegFormer, {metric_dict[metric2]}')
        axs[2, 0].set_xlim([0, 1])
        axs[2, 0].set_facecolor('0.9')
        axs[2, 1].scatter(segformer_roboflow_morph[metric_dict[metric1]], segformer_roboflow_morph[metric_dict[metric2]], s=10, color='tab:orange')
        axs[2, 1].scatter(segformer_goNorth_morph[metric_dict[metric1]], segformer_goNorth_morph[metric_dict[metric2]], s=10, color='tab:purple')
        axs[2, 1].set_xlabel(metric_dict[metric1])
        axs[2, 1].set_xlim([0, 1])
        axs[2, 1].set_facecolor('0.9')
        axs[2, 2].scatter(segformer_roboflow_otsu[metric_dict[metric1]], segformer_roboflow_otsu[metric_dict[metric2]], s=10, color='tab:green')
        axs[2, 2].scatter(segformer_goNorth_otsu[metric_dict[metric1]], segformer_goNorth_otsu[metric_dict[metric2]], s=10, color='tab:brown')
        axs[2, 2].set_xlim([0, 1])
        axs[2, 2].set_facecolor('0.9')

        plt.tight_layout()
        plt.show()
        plt.close()
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        #axs[0].scatter(unet_roboflow_raw[metric_dict[metric1]], unet_roboflow_raw[metric_dict[metric2]]+unet_roboflow_raw[metric_dict[metric3]], s=10, color='tab:blue', label='Roboflow')
        #axs[0].scatter(unet_goNorth_raw[metric_dict[metric1]], unet_goNorth_raw[metric_dict[metric2]]+unet_goNorth_raw[metric_dict[metric3]], s=10, color='tab:red', label='GoNorth')
        axs[0].scatter(unet_roboflow_raw[metric_dict[metric1]], unet_roboflow_raw[metric_dict[metric2]], s=10, color='tab:blue', label='Roboflow')
        axs[0].scatter(unet_goNorth_raw[metric_dict[metric1]], unet_goNorth_raw[metric_dict[metric2]], s=10, color='tab:red', label='GoNorth')
        # axs[0].set_ylabel(metric_dict[metric2], fontsize=12)
        axs[0].set_ylabel('SIPP Prediction', fontsize=12)
        # axs[0].set_xlabel('SIPP Label', fontsize=12)
        axs[0].set_title('Raw-trained U-Net', fontsize=12)
        axs[0].set_xlim([0, 1.05])
        #axs[0].set_ylim([0, 1.1])
        axs[0].set_facecolor('0.9')
        axs[0].legend(fontsize=12)


        #axs[1].scatter(unet_roboflow_morph[metric_dict[metric1]], unet_roboflow_morph[metric_dict[metric2]]+unet_roboflow_morph[metric_dict[metric3]], s=10, color='tab:orange', label='Roboflow')
        #axs[1].scatter(unet_goNorth_morph[metric_dict[metric1]], unet_goNorth_morph[metric_dict[metric2]]+unet_goNorth_morph[metric_dict[metric3]], s=10, color='tab:purple', label='GoNorth')
        axs[1].scatter(unet_roboflow_morph[metric_dict[metric1]], unet_roboflow_morph[metric_dict[metric2]], s=10, color='tab:orange', label='Roboflow')
        axs[1].scatter(unet_goNorth_morph[metric_dict[metric1]], unet_goNorth_morph[metric_dict[metric2]], s=10, color='tab:purple', label='GoNorth')
        axs[1].set_xlabel('SIPP Label', fontsize=12)
        axs[1].set_title('Morph-trained U-Net', fontsize=12)
        axs[1].set_xlim([0, 1.05])
        axs[1].set_facecolor('0.9')
        axs[1].legend(fontsize=12)

        #axs[2].scatter(unet_roboflow_otsu[metric_dict[metric1]], unet_roboflow_otsu[metric_dict[metric2]]+unet_roboflow_otsu[metric_dict[metric3]], s=10, color='tab:green', label='Roboflow')
        #axs[2].scatter(unet_goNorth_otsu[metric_dict[metric1]], unet_goNorth_otsu[metric_dict[metric2]]+unet_goNorth_otsu[metric_dict[metric3]], s=10, color='tab:brown', label='GoNorth')
        axs[2].scatter(unet_roboflow_otsu[metric_dict[metric1]], unet_roboflow_otsu[metric_dict[metric2]], s=10, color='tab:green', label='Roboflow')
        axs[2].scatter(unet_goNorth_otsu[metric_dict[metric1]], unet_goNorth_otsu[metric_dict[metric2]], s=10, color='tab:brown', label='GoNorth')
        axs[2].set_title('Otsu-trained U-Net', fontsize=12)
        axs[2].set_xlim([0, 1.05])
        axs[2].set_facecolor('0.9')
        axs[2].legend(fontsize=12)

        plt.tight_layout()
        plt.show()
        plt.close()
        exit()

    
    



if __name__ == '__main__':

    model = sys.argv[1]
    metric1 = sys.argv[2]
    metric2 = sys.argv[3]
    mode = sys.argv[4]

    spot_correlations(model, metric1, metric2, mode)



    

