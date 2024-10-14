#!/usr/bin/env bash

echo "This script is to evaluate all models on all datasets"

models=("unet" "deeplabv3plus" "segformer")
eval_datasets=("goNorth roboflow")
train_datasets=("raw" "morph" "otsu")

for model in ${models[@]};
do
    for eval_dataset in ${eval_datasets[@]};
    do
        for train_dataset in ${train_datasets[@]};
        do
            echo "evaluating $model trained on $train_dataset on labelled dataset $eval_dataset"
            python evaluate_labelledset.py $model $eval_dataset $train_dataset
        done     
    done
done