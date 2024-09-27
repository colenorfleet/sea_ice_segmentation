#!/usr/bin/env bash

echo "This script is to train all models on all datasets"

echo "How many epochs?"
read epochs

echo "What batch size?"
read batch_size

models=("unet" "deeplabv3plus" "segformer")
datasets=("raw" "morph" "otsu")

for model in ${models[@]};
do
    for dataset in ${datasets[@]};
        do
            echo "Training $model on $dataset, for $epochs epochs, with a batch size of $batch_size"

            python train.py $model $dataset $epochs $batch_size

    done
done
