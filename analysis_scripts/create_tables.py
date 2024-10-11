
import os
import sys
import csv
import pandas as pd
from datetime import date

output_path = '/home/cole/Pictures/thesis_report/tables/'


def create_tables(mode='evaluation_model'):

    dir_path = '/home/cole/Documents/NTNU/sea_ice_segmentation/'
    models = ['unet', 'deeplabv3plus', 'segformer']
    datasets = ['raw', 'morph', 'otsu']

    if mode=='training_model':
        # create the training tables

        # Folder Structure
        # output
            # model_name
                # dataset_name
                    # training_logs.csv

        ex_file = pd.read_csv(f'{dir_path}/output/deeplabv3plus/morph/training_logs.csv')
        num_epochs_ex = len(ex_file['Epoch'])
        csv_file = os.path.abspath(os.path.join(output_path, "training_statistics_by_model_{}.csv".format(date.today())))
        csv_header = ['Model', 'Dataset', 'Train Loss', 'Val Loss', 'Train IOU', 'Val IOU', 'Train DICE Score', 'Val DICE Score']

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
                            f"{avgs['Avg Train DICE']:0.2f}",
                            f"{avgs['Avg Val DICE']:0.2f}"
                        ]
                    )

    elif mode=='training_dataset':

        ex_file = pd.read_csv(f'{dir_path}/output/deeplabv3plus/morph/training_logs.csv')
        num_epochs_ex = len(ex_file['Epoch'])
        csv_file = os.path.abspath(os.path.join(output_path, "training_statistics_by_dataset_{}.csv".format(date.today())))
        csv_header = ['Dataset', 'Model', 'Train Loss', 'Val Loss', 'Train IOU', 'Val IOU', 'Train DICE Score', 'Val DICE Score']

        with open(csv_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for dataset in datasets:
                for model in models:
                    training_stats = pd.read_csv(f'{dir_path}/output/{model}/{dataset}/training_logs.csv')
                    assert num_epochs_ex == len(training_stats['Epoch']), "Number of epochs do not match"

                    avgs = training_stats.mean().round(2)

                    csv_writer.writerow(
                        [
                            dataset,
                            model,
                            f"{avgs['Avg BCE Train Loss']:0.2f}",
                            f"{avgs['Avg BCE Val Loss']:0.2f}",
                            f"{avgs['Avg Train IOU']:0.2f}",
                            f"{avgs['Avg Val IOU']:0.2f}",
                            f"{avgs['Avg Train DICE']:0.2f}",
                            f"{avgs['Avg Val DICE']:0.2f}"
                        ]
                    )     
        


    elif mode=='evaluation_model':
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
        csv_file = os.path.abspath(os.path.join(output_path, "test_set_performance_by_model_{}.csv".format(date.today())))

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
                            f"{avgs['DICE']:0.2f}",
                            f"{avgs['Pixel Accuracy']:0.2f}",
                            f"{avgs['Precision']:0.2f}",
                            f"{avgs['Recall']:0.2f}",
                            int(avgs['Number True Positive']),
                            int(avgs['Number False Positive']),
                            int(avgs['Number True Negative']),
                            int(avgs['Number False Negative']),
                            f"{avgs['SIC Label']:0.2f}",
                            f"{avgs['SIC Pred']:0.2f}"
                        ]
                    )

    elif mode=='evaluation_dataset':

        ex_file = pd.read_csv(f'{dir_path}/test_data_output/deeplabv3plus/morph/evaluation_scores.csv')

        csv_header = list(ex_file.columns)
        csv_header.remove('Sample')
        csv_header[0:0] = ['Dataset', 'Model']
        csv_file = os.path.abspath(os.path.join(output_path, "test_set_performance_by_dataset_{}.csv".format(date.today())))

        with open(csv_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for dataset in datasets:
                for model in models:
                    evaluation_scores = pd.read_csv(f'{dir_path}/test_data_output/{model}/{dataset}/evaluation_scores.csv')

                    avgs = evaluation_scores.mean().round(2)

                    csv_writer.writerow(
                        [
                            dataset,
                            model,
                            f"{avgs['BCE Loss']:0.2f}", 
                            f"{avgs['IOU']:0.2f}",
                            f"{avgs['DICE']:0.2f}",
                            f"{avgs['Pixel Accuracy']:0.2f}",
                            f"{avgs['Precision']:0.2f}",
                            f"{avgs['Recall']:0.2f}",
                            int(avgs['Number True Positive']),
                            int(avgs['Number False Positive']),
                            int(avgs['Number True Negative']),
                            int(avgs['Number False Negative']),
                            f"{avgs['SIC Label']:0.2f}",
                            f"{avgs['SIC Pred']:0.2f}"
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
        csv_header = ['Model', 'Dataset', 'Train Loss', 'Val Loss', 'Train IOU', 'Val IOU', 'Train DICE Score', 'Val DICE Score']

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
                            f"{avgs['Avg Train DICE']:0.2f}",
                            f"{avgs['Avg Val DICE']:0.2f}"
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
                            f"{avgs['DICE']:0.2f}",
                            f"{avgs['Pixel Accuracy']:0.2f}",
                            f"{avgs['Precision']:0.2f}",
                            f"{avgs['Recall']:0.2f}",
                            int(avgs['Number True Positive']),
                            int(avgs['Number False Positive']),
                            int(avgs['Number True Negative']),
                            int(avgs['Number False Negative']),
                            f"{avgs['SIC Label']:0.2f}",
                            f"{avgs['SIC Pred']:0.2f}"
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
        csv_header = ['Dataset', 'Model', 'Fold', 'Train Loss', 'Val Loss', 'Train IOU', 'Val IOU', 'Train DICE Score', 'Val DICE Score']

        with open(csv_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for dataset in datasets:
                for model in models:
                    for fold in range(5):

                        training_stats = pd.read_csv(f'{dir_path}/cv_output/{model}/{dataset}/training_logs_fold_{fold}.csv')
                        assert num_epochs_ex == len(training_stats['Epoch']), "Number of epochs do not match"

                        last_row = training_stats.tail(1).round(2)

                        csv_writer.writerow(
                            [
                                dataset,
                                model,
                                fold+1,
                                f"{last_row['Avg BCE Train Loss'].round(2).item()}",
                                f"{last_row['Avg BCE Val Loss'].round(2).item()}",
                                f"{last_row['Avg Train IOU'].round(2).item()}",
                                f"{last_row['Avg Val IOU'].round(2).item()}",
                                f"{last_row['Avg Train DICE'].round(2).item()}",
                                f"{last_row['Avg Val DICE'].round(2).item()}"
                            ]
                        )



    else:
        print("Invalid mode")
        return
    

    


if __name__ == '__main__':

    mode = sys.argv[1]
    create_tables(mode=mode)