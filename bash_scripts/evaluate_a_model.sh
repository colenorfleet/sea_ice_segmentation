#!/usr/bin/env bash

# Run all models

echo "What mode would you like to run? This is only for one model at at a time"
read mode 

echo "Which model(s) would you like to $mode?"
read model

echo "Running $model, on which dataset? (put all)"
read dataset


if [ $mode == "both" ]; then

    echo "Running $model on $dataset, for how many epochs?"
    read epochs

    echo "Running $model on $dataset for $epochs epochs, with what batch size?"
    read batch_size

    echo "Running $model for $epochs epochs, on $dataset, with a batch size of $batch_size"

    echo "ok, training and evaluating model $model on all datasets for $epochs epochs with a batch size of $batch_size"

    echo "Step 1: Running $model on raw"
    python train.py $model raw $epochs $batch_size

    echo "Step 2: Running $model on morph"
    python train.py $model morph $epochs $batch_size

    echo "Step 3: Running $model on otsu"
    python train.py $model otsu $epochs $batch_size

    echo "Step 4: Evaluating $model on raw"
    python evaluate_testset.py $model raw

    echo "Step 5: Evaluating $model on morph"
    python evaluate_testset.py $model morph

    echo "Step 6: Evaluating $model on otsu"
    python evaluate_testset.py $model otsu

elif [ $mode == "train" ]; then

    echo "Running $model on $dataset, for how many epochs?"
    read epochs

    echo "Running $model on $dataset for $epochs epochs, with what batch size?"
    read batch_size

    echo "Running $model for $epochs epochs, on $dataset, with a batch size of $batch_size"

    echo "ok, training only for $model"

    echo "Step 1: Running $model on raw"
    python train.py $model raw $epochs $batch_size

    echo "Step 2: Running $model on morph"
    python train.py $model morph $epochs $batch_size

    echo "Step 3: Running $model on otsu"
    python train.py $model otsu $epochs $batch_size

elif [ $mode == "evaluate" ]; then

    echo "ok, evaluating only for $model"

    echo "Step 1: Evaluating $model on raw"
    python evaluate_testset.py $model raw

    echo "Step 2: Evaluating $model on morph"
    python evaluate_testset.py $model morph

    echo "Step 3: Evaluating $model on otsu"
    python evaluate_testset.py $model otsu

fi