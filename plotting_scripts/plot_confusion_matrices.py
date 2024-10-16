import os 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_condusion_matrices(mode='predictions'):

    if mode == 'predictions':

        for model in ['unet', 'deeplabv3plus', 'segformer']:

            raw_roboflow = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/{model}/roboflow/raw/evaluation_scores.csv')
            raw_gonorth = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/{model}/goNorth/raw/evaluation_scores.csv')

            morph_roboflow = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/{model}/roboflow/morph/evaluation_scores.csv')
            morph_gonorth = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/{model}/goNorth/morph/evaluation_scores.csv')

            otsu_roboflow = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/{model}/roboflow/otsu/evaluation_scores.csv')
            otsu_gonorth = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output/{model}/goNorth/otsu/evaluation_scores.csv')

            raw_both = pd.concat([raw_roboflow, raw_gonorth])
            morph_both = pd.concat([morph_roboflow, morph_gonorth])
            otsu_both = pd.concat([otsu_roboflow, otsu_gonorth])

            raw_tp = raw_both['Number True Positive'].mean()
            raw_fp = raw_both['Number False Positive'].mean()
            raw_tn = raw_both['Number True Negative'].mean()
            raw_fn = raw_both['Number False Negative'].mean()

            morph_tp = morph_both['Number True Positive'].mean()
            morph_fp = morph_both['Number False Positive'].mean()
            morph_tn = morph_both['Number True Negative'].mean()
            morph_fn = morph_both['Number False Negative'].mean()

            otsu_tp = otsu_both['Number True Positive'].mean()
            otsu_fp = otsu_both['Number False Positive'].mean()
            otsu_tn = otsu_both['Number True Negative'].mean()
            otsu_fn = otsu_both['Number False Negative'].mean()


            raw_confusion_matrix = np.array([[raw_tn, raw_fp], [raw_fn, raw_tp]])
            morph_confusion_matrix = np.array([[morph_tn, morph_fp], [morph_fn, morph_tp]])
            otsu_confusion_matrix = np.array([[otsu_tn, otsu_fp], [otsu_fn, otsu_tp]])

            fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

            sns.heatmap(raw_confusion_matrix/np.sum(raw_confusion_matrix), annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax[0])
            ax[0].set_title('Raw-trained')
            # ax[0].set_xlabel('Predicted')
            ax[0].set_ylabel('Labelled')

            sns.heatmap(morph_confusion_matrix/np.sum(morph_confusion_matrix), annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax[1])
            ax[1].set_title('Morph-trained')
            ax[1].set_xlabel('Predicted')
            # ax[1].set_ylabel('Actual')

            sns.heatmap(otsu_confusion_matrix/np.sum(otsu_confusion_matrix), annot=True, fmt=".2%", cmap="Blues", ax=ax[2])
            ax[2].set_title('Otsu-trained')
            # ax[2].set_xlabel('Predicted')
            # ax[2].set_ylabel('Actual')
            
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'/home/cole/Pictures/thesis_report/test_set_statistics/confusion_matrices/{model}_prediction_confusion_matrices.png')

    elif mode == 'dataset_evaluation':

        robowflow_df = pd.read_csv('/home/cole/Pictures/thesis_report/labelled_evaluation/labelled_metrics.csv')
        gonorth_df = pd.read_csv('/home/cole/Pictures/thesis_report/labelled_evaluation/GoNorth_labelled_metrics.csv')

        df = pd.concat([robowflow_df, gonorth_df])
        df.set_index('Dataset', inplace=True)

        raw_tp, raw_fp, raw_tn, raw_fn = df.loc['raw', 'Number True Positive'].mean(), df.loc['raw', 'Number False Positive'].mean(), df.loc['raw', 'Number True Negative'].mean(), df.loc['raw', 'Number False Negative'].mean()
        morph_tp, morph_fp, morph_tn, morph_fn = df.loc['morph', 'Number True Positive'].mean(), df.loc['morph', 'Number False Positive'].mean(), df.loc['morph', 'Number True Negative'].mean(), df.loc['morph', 'Number False Negative'].mean()
        otsu_tp, otsu_fp, otsu_tn, otsu_fn = df.loc['otsu', 'Number True Positive'].mean(), df.loc['otsu', 'Number False Positive'].mean(), df.loc['otsu', 'Number True Negative'].mean(), df.loc['otsu', 'Number False Negative'].mean()

        raw_confusion_matrix = np.array([[raw_tn, raw_fp], [raw_fn, raw_tp]])
        morph_confusion_matrix = np.array([[morph_tn, morph_fp], [morph_fn, morph_tp]])
        otsu_confusion_matrix = np.array([[otsu_tn, otsu_fp], [otsu_fn, otsu_tp]])

        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        sns.heatmap(raw_confusion_matrix/np.sum(raw_confusion_matrix), annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax[0])
        ax[0].set_title('Raw Dataset')
        # ax[0].set_xlabel('Preprocessed')
        ax[0].set_ylabel('Labelled')

        sns.heatmap(morph_confusion_matrix/np.sum(morph_confusion_matrix), annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax[1])
        ax[1].set_title('Morph Dataset')
        ax[1].set_xlabel('Preprocessed')
        # ax[1].set_ylabel('Labelled')

        sns.heatmap(otsu_confusion_matrix/np.sum(otsu_confusion_matrix), annot=True, fmt=".2%", cmap="Blues", ax=ax[2])
        ax[2].set_title('Otsu Dataset')
        # ax[2].set_xlabel('Preprocessed')
        # ax[2].set_ylabel('Labelled')

        plt.tight_layout()
        # plt.show() 
        plt.savefig('/home/cole/Pictures/thesis_report/test_set_statistics/confusion_matrices/dataset_confusion_matrices.png')


    

if __name__ == '__main__':
    mode = sys.argv[1]
    plot_condusion_matrices(mode=mode)




'''
models = {'unet', 'deeplabv3plus', 'segformer'}

    for model in models:

        raw_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/raw/evaluation_scores.csv')
        morph_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/morph/evaluation_scores.csv')
        otsu_stats = pd.read_csv(f'/home/cole/Documents/NTNU/sea_ice_segmentation/test_data_output/{model}/otsu/evaluation_scores.csv')

        datasets = {'raw', 'morph', 'otsu'}
        categories = {'TP', 'FP', 'TN', 'FN'}

        # order: TP, FP, TN, FN
        averages = {
            'raw': (raw_stats['Number True Positive'].mean(), 
                    raw_stats['Number False Positive'].mean(), 
                    raw_stats['Number True Negative'].mean(),
                    raw_stats['Number False Negative'].mean()),
            'morph': (morph_stats['Number True Positive'].mean(),
                      morph_stats['Number False Positive'].mean(),
                      morph_stats['Number True Negative'].mean(),
                      morph_stats['Number False Negative'].mean()),
            'otsu': (otsu_stats['Number True Positive'].mean(),
                     otsu_stats['Number False Positive'].mean(),
                     otsu_stats['Number True Negative'].mean(),
                     otsu_stats['Number False Negative'].mean()),
        }
           
        x = np.arange(len(categories))
        width = 0.15
        
        fig, ax = plt.subplots(layout='constrained', figsize=(20, 10))
        multiplier = 0
        max_value = 0

        for attribute, measurement in averages.items():
            offset = width * multiplier
            if max(measurement) > max_value:
                max_value = max(measurement)
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_xlabel('Pixel Classification')
        ax.set_ylabel('Number of Pixels')
        ax.set_title(f'Average Confusion Matrix Values for {model}')
        ax.set_xticks(x + width, ['True Positive', 'False Positive', 'True Negative', 'False Negative'])
        ax.legend(loc='upper left', ncols=4)
        ax.set_ylim(0, 185000)


        plt.savefig(f'/home/cole/Pictures/thesis_report/test_set_statistics/confusion_matrices/{model}_confusion_matrix.png')
        plt.close()
'''