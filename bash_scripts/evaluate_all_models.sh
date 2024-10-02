#!/usr/bin/env bash

echo "This script is to evaluate all models on all datasets"

models=("unet" "deeplabv3plus" "segformer")
datasets=("raw" "morph" "otsu")

for model in ${models[@]};
do
    for dataset in ${datasets[@]};
        do
            echo "evaluating $model on $dataset"

            python evaluate_testset.py $model $dataset
            
    done
done