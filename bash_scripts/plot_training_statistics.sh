#!/usr/bin/env bash

echo "Visualizing all training statistics, make sure to check output directory in script"

models=("unet" "deeplabv3plus" "segformer")
datasets=("raw" "morph" "otsu")
metrics=("loss" "iou" "f1")

for model in "${models[@]}";
do
    for metric in "${metrics[@]}";
    do
        python plotting_scripts/view_training_statistics.py $model all $metric save=True
    done
done


for dataset in "${datasets[@]}";
do
    for metric in "${metrics[@]}";
    do
        python plotting_scripts/view_training_statistics.py all $dataset $metric save=True
    done
done