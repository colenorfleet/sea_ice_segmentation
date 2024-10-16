import os
import pandas as pd
import numpy as np
import csv



output_path = '/home/cole/Pictures/thesis_report/labelled_evaluation'
GoNorth = pd.read_csv(os.path.join(output_path, 'dataset_eval/GoNorth_labelled_metrics.csv'))
Roboflow = pd.read_csv(os.path.join(output_path, 'dataset_eval/labelled_metrics.csv'))

df_dict = {'GoNorth': GoNorth, 'Roboflow': Roboflow}

# index by dataset

GoNorth.set_index('Dataset', inplace=True)
Roboflow.set_index('Dataset', inplace=True)

csv_file = os.path.abspath(os.path.join(output_path, "both_labelled_dataset_metrics.csv"))
csv_header = ['Subset', 'Dataset', 'IOU', 'DICE', 'Pixel Accuracy', 'Precision', 'Recall', 'SIC Manual', 'SIC Processed']


with open(csv_file, mode='w') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(csv_header)

    for dataframe in ['GoNorth', 'Roboflow']:
        df = df_dict[dataframe]

        for dataset in ['raw', 'morph', 'otsu']:

            iou = df.loc[dataset, 'IOU'].mean().round(2)
            dice = df.loc[dataset, 'DICE'].mean().round(2)
            pixel_acc = df.loc[dataset, 'Pixel Accuracy'].mean().round(2)
            precision = df.loc[dataset, 'Precision'].mean().round(2)
            recall = df.loc[dataset, 'Recall'].mean().round(2)
            sic_manual = df.loc[dataset, 'SIC Manual'].mean().round(2)
            sic_processed = df.loc[dataset, 'SIC Processed'].mean().round(2)

            csv_writer.writerow([
                dataframe, 
                dataset,
                iou,
                dice,
                pixel_acc,
                precision,
                recall,
                sic_manual,
                sic_processed
            ])
