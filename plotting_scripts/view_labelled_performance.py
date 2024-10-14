
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


input_dict = {'loss': 'BCE Loss',
            'iou': 'IOU',
            'precision': 'Precision',
            'recall': 'Recall',
            'accuracy': 'Pixel Accuracy',
            'dice': 'DICE',
            'sic': 'SIC Label'
            }

output_dir = '/home/cole/Documents/NTNU/sea_ice_segmentation/labelled_output'
plot_dir = '/home/cole/Pictures/thesis_report/labelled_evaluation/plots'

def view_labelled_performance(model_name, metric='iou'):

    fig = plt.figure(figsize=(12, 6))

    goNorth_raw = pd.read_csv(os.path.join(output_dir, model_name, 'goNorth', 'raw', 'evaluation_scores.csv'))
    goNorth_morph = pd.read_csv(os.path.join(output_dir, model_name, 'goNorth', 'morph', 'evaluation_scores.csv'))
    goNorth_otsu = pd.read_csv(os.path.join(output_dir, model_name, 'goNorth', 'otsu', 'evaluation_scores.csv'))

    roboflow_raw = pd.read_csv(os.path.join(output_dir, model_name, 'roboflow', 'raw', 'evaluation_scores.csv'))
    roboflow_morph = pd.read_csv(os.path.join(output_dir, model_name, 'roboflow', 'morph', 'evaluation_scores.csv'))
    roboflow_otsu = pd.read_csv(os.path.join(output_dir, model_name, 'roboflow', 'otsu', 'evaluation_scores.csv'))

    data_length = len(roboflow_raw)

    goNorth_raw.index += data_length
    goNorth_morph.index += data_length
    goNorth_otsu.index += data_length

    both_raw = pd.concat([roboflow_raw, goNorth_raw])
    both_morph = pd.concat([roboflow_morph, goNorth_morph])
    both_otsu = pd.concat([roboflow_otsu, goNorth_otsu])

    raw_mean = both_raw[input_dict[metric]].mean().round(2)
    morph_mean = both_morph[input_dict[metric]].mean().round(2)
    otsu_mean = both_otsu[input_dict[metric]].mean().round(2)

    plt.plot(both_raw[input_dict[metric]], label='raw, mean: ' + str(raw_mean))
    plt.plot(both_morph[input_dict[metric]], label='morph, mean: ' + str(morph_mean))
    plt.plot(both_otsu[input_dict[metric]], label='otsu, mean: ' + str(otsu_mean))

    plt.fill_between(both_raw['SIC Label'].index, both_raw['SIC Label'], 0, color='gray', alpha=0.2)

    plt.vlines([data_length], 0, 1.25, color='black', linestyle='-')
    plt.text((data_length/2)-20, 1.05, 'Roboflow', fontsize=14)
    plt.text(data_length*1.5, 1.05, 'goNorth', fontsize=14)

    plt.tight_layout()
    plt.grid()
    # plt.ylabel(metric, size=14)
    plt.ylim(0, 1.25)
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(plot_dir, f'{model_name}_{metric}.png'))
    plt.close()



if __name__ == "__main__":

    model_name = sys.argv[1]
    metric = sys.argv[2]

    view_labelled_performance(model_name, metric)