
import os
import sys
import csv
import pandas as pd
from datetime import date

output_path = '/home/cole/Pictures/thesis_report/tables/'


def create_tables(mode='evaluation'):

    dir_path = '/home/cole/Documents/NTNU/sea_ice_segmentation/'
    models = ['unet', 'deeplabv3plus', 'segformer']
    datasets = ['raw', 'morph', 'otsu']

    if mode=='training':
        # create the training tables

        # Folder Structure
        # output
            # model_name
                # dataset_name
                    # training_logs.csv

        ex_file = pd.read_csv(f'{dir_path}/output/deeplabv3plus/morph/training_logs.csv')
        num_epochs_ex = len(ex_file['Epoch'])
        csv_file = os.path.abspath(os.path.join(output_path, "training_statistics_{}.csv".format(date.today())))
        csv_header = ['Model', 'Dataset', 'Train Loss', 'Val Loss', 'Train IOU', 'Val IOU', 'Train F1 Score', 'Val F1 Score']

        with open(csv_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for model in models:
                for dataset in datasets:
                    training_stats = pd.read_csv(f'{dir_path}/output/{model}/{dataset}/training_logs.csv')
                    assert num_epochs_ex == len(training_stats['Epoch']), "Number of epochs do not match"

                    avgs = training_stats.mean().round(2)

                    csv_writer.writerow(
                        [
                            model,
                            dataset,
                            f"{avgs['Avg BCE Train Loss']:0.2f}",
                            f"{avgs['Avg BCE Val Loss']:0.2f}",
                            f"{avgs['Avg Train IOU']:0.2f}",
                            f"{avgs['Avg Val IOU']:0.2f}",
                            f"{avgs['Avg Train F1']:0.2f}",
                            f"{avgs['Avg Val F1']:0.2f}"
                        ]
                    )     
        


    elif mode=='evaluation':
        # create the evaluation tables

        # Folder Structure
        # test_data_output
            # model_name
                # dataset_name
                    # evaluation_scores.csv

        ex_file = pd.read_csv(f'{dir_path}/test_data_output/deeplabv3plus/morph/evaluation_scores.csv')

        csv_header = list(ex_file.columns)
        csv_header.remove('Sample')
        csv_header[0:0] = ['Model', 'Dataset']
        csv_file = os.path.abspath(os.path.join(output_path, "test_set_performance_{}.csv".format(date.today())))

        with open(csv_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for model in models:
                for dataset in datasets:
                    evaluation_scores = pd.read_csv(f'{dir_path}/test_data_output/{model}/{dataset}/evaluation_scores.csv')

                    avgs = evaluation_scores.mean().round(2)

                    csv_writer.writerow(
                        [
                            model,
                            dataset,
                            f"{avgs['BCE Loss']:0.2f}", 
                            f"{avgs['IOU']:0.2f}",
                            f"{avgs['Dice Coefficient']:0.2f}",
                            f"{avgs['Pixel Accuracy']:0.2f}",
                            f"{avgs['Foreground Accuracy']:0.2f}",
                            f"{avgs['Background Accuracy']:0.2f}",
                            f"{avgs['False Negative Rate']:0.2f}",
                            f"{avgs['SIC Label']:0.2f}",
                            f"{avgs['SIC Prediction']:0.2f}",
                            int(avgs['Number True Positive']),
                            int(avgs['Number False Positive']),
                            int(avgs['Number True Negative']),
                            int(avgs['Number False Negative']),
                            f"{avgs['F1 Score']:0.2f}"
                        ]
                    )


    elif mode=='all_dataset_training':

        # Folder Structure
        # all_dataset_output
            # model_name
                # best_ice_seg_model.pth
                # training_logs.csv

        ex_file = pd.read_csv(f'{dir_path}/all_dataset_output/deeplabv3plus/training_logs.csv')
        num_epochs_ex = len(ex_file['Epoch'])
        csv_file = os.path.abspath(os.path.join(output_path, "all_dataset_training_statistics_{}.csv".format(date.today())))
        csv_header = ['Model', 'Dataset', 'Train Loss', 'Val Loss', 'Train IOU', 'Val IOU', 'Train F1 Score', 'Val F1 Score']

        with open(csv_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for model in models:
                for dataset in datasets:

                    training_stats = pd.read_csv(f'{dir_path}/all_dataset_output/{model}/training_logs.csv')
                    assert num_epochs_ex == len(training_stats['Epoch']), "Number of epochs do not match"

                    training_stats.set_index('Dataset', inplace=True)

                    avgs = training_stats.loc[dataset].mean().round(2)

                    csv_writer.writerow(
                        [
                            model,
                            dataset,
                            f"{avgs['Avg BCE Train Loss']:0.2f}",
                            f"{avgs['Avg BCE Val Loss']:0.2f}",
                            f"{avgs['Avg Train IOU']:0.2f}",
                            f"{avgs['Avg Val IOU']:0.2f}",
                            f"{avgs['Avg Train F1']:0.2f}",
                            f"{avgs['Avg Val F1']:0.2f}"
                        ]
                    )
    
    elif mode=='all_dataset_evaluation':

        ex_file = pd.read_csv(f'{dir_path}/test_data_output/all_dataset_train/deeplabv3plus/morph/evaluation_scores.csv')

        csv_header = list(ex_file.columns)
        csv_header.remove('Sample')
        csv_header[0:0] = ['Model', 'Dataset']
        csv_file = os.path.abspath(os.path.join(output_path, "all_dataset_train_test_set_performance_{}.csv".format(date.today())))

        with open(csv_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for model in models:
                for dataset in datasets:
                    evaluation_scores = pd.read_csv(f'{dir_path}/test_data_output/all_dataset_train/{model}/{dataset}/evaluation_scores.csv')

                    avgs = evaluation_scores.mean().round(2)

                    csv_writer.writerow(
                        [
                            model,
                            dataset,
                            f"{avgs['BCE Loss']:0.2f}", 
                            f"{avgs['IOU']:0.2f}",
                            f"{avgs['Dice Coefficient']:0.2f}",
                            f"{avgs['Pixel Accuracy']:0.2f}",
                            f"{avgs['Foreground Accuracy']:0.2f}",
                            f"{avgs['Background Accuracy']:0.2f}",
                            f"{avgs['False Negative Rate']:0.2f}",
                            f"{avgs['SIC Label']:0.2f}",
                            f"{avgs['SIC Prediction']:0.2f}",
                            int(avgs['Number True Positive']),
                            int(avgs['Number False Positive']),
                            int(avgs['Number True Negative']),
                            int(avgs['Number False Negative']),
                            f"{avgs['F1 Score']:0.2f}"
                        ]
                    )

        


    elif mode=='cross_validation_training':
        # create the cross validation tables

        # Folder Structure
        # cv_output
            # model_name
                # dataset_name
                    # best_ice_seg_model_fold_{0-4}.pth
                    # training_logs_fold_{0-4}.csv

        ex_file = pd.read_csv(f'{dir_path}/cv_output/deeplabv3plus/morph/training_logs_fold_3.csv')
        num_epochs_ex = len(ex_file['Epoch'])
        csv_file = os.path.abspath(os.path.join(output_path, "cv_training_statistics_{}.csv".format(date.today())))
        csv_header = ['Model', 'Dataset', 'Fold', 'Train Loss', 'Val Loss', 'Train IOU', 'Val IOU', 'Train F1 Score', 'Val F1 Score']

        with open(csv_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for model in models:
                for dataset in datasets:
                    for fold in range(5):

                        training_stats = pd.read_csv(f'{dir_path}/cv_output/{model}/{dataset}/training_logs_fold_{fold}.csv')
                        assert num_epochs_ex == len(training_stats['Epoch']), "Number of epochs do not match"

                        avgs = training_stats.mean().round(2)

                        csv_writer.writerow(
                            [
                                model,
                                dataset,
                                fold+1,
                                f"{avgs['Avg BCE Train Loss']:0.2f}",
                                f"{avgs['Avg BCE Val Loss']:0.2f}",
                                f"{avgs['Avg Train IOU']:0.2f}",
                                f"{avgs['Avg Val IOU']:0.2f}",
                                f"{avgs['Avg Train F1']:0.2f}",
                                f"{avgs['Avg Val F1']:0.2f}"
                            ]
                        )



    else:
        print("Invalid mode")
        return
    

    


if __name__ == '__main__':

    mode = sys.argv[1]
    create_tables(mode=mode)