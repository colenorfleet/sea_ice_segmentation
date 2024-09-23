#!/usr/bin/env bash

echo "This script is to run a models on all datasets"

echo "What model?"
read model

echo "How many epochs?"
read epochs

echo "What batch size?"
read batch_size

# models=`ls /home/cole/Documents/NTNU/sea_ice_segmentation/output`
datasets=`ls /home/cole/Documents/NTNU/datasets | grep -v ".txt"`

for dataset in $datasets;
    do
        echo "Running $model on $dataset, for $epochs epochs, with a batch size of $batch_size"

        python train.py $model $dataset $epochs $batch_size

done

#for model in $models;
#do
#    for dataset in $datasets;
#    do
#        echo "Running $model on $dataset, for $epochs epochs, with a batch size of $batch_size"
#
#        python train.py $model $dataset $epochs $batch_size
#
#    done

#done
