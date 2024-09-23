#!/usr/bin/env bash

# Run all models

echo "Which model(s) would you like to run?"
read model

echo "Running $model, on which dataset?"
read dataset

echo "Running $model on $dataset, for how many epochs?"
read epochs

echo "Running $model on $dataset for $epochs epochs, with what batch size?"
read batch_size

echo "Running $model for $epochs epochs, on $dataset, with a batch size of $batch_size"


if [ $model == "all" ]; then
    echo "Running all models"

    models = 

    if [ $dataset == "all" ]; then
        echo "Running all models on all datasets for $epochs epochs with a batch size of $batch_size"

        echo "Step 1: Running unet on raw"
        python train.py unet_brain raw $epochs $batch_size

        echo "Step 2: Running unet on morph"
        python train.py unet_brain morph $epochs $batch_size

        echo "Step 3: Running unet on otsu"
        python train.py unet_brain otsu $epochs $batch_size

        echo "Step 4: Running deeplabv3 on raw"
        python train.py deeplabv3 raw $epochs $batch_size

        echo "Step 5: Running deeplabv3 on morph"
        python train.py deeplabv3 morph $epochs $batch_size

        echo "Step 6: Running deeplabv3 on otsu"
        python train.py deeplabv3 otsu $epochs $batch_size

        echo "Step 7: Running dinov2 on raw"
        python train.py dinov2 raw $epochs $batch_size

        echo "Step 8: Running dinov2 on morph"
        python train.py dinov2 morph $epochs $batch_size

        echo "Step 9: Running dinov2 on otsu"
        python train.py dinov2 otsu $epochs $batch_size

        echo "Step 10: Running pspnet on raw"
        python train.py pspnet raw $epochs $batch_size

        echo "Step 11: Running pspnet on morph"
        python train.py pspnet morph $epochs $batch_size

        echo "Step 12: Running pspnet on otsu"
        python train.py pspnet otsu $epochs $batch_size

    elif [ $dataset != "all" ]; then
        echo "Running all models on dataset $dataset for $epochs epochs with a batch size of $batch_size"

        echo "Step 1: Running unet on $dataset"
        python train.py unet $dataset $epochs $batch_size

        echo "Step 2: Running deeplabv3 on $dataset"
        python train.py deeplabv3 $dataset $epochs $batch_size

        echo "Step 3: Running dinov2 on $dataset"
        python train.py dinov2 $dataset $epochs $batch_size

    fi

elif [ $model != "all" ]; then
    echo "Running $model"

    if [ $dataset == "all" ]; then
        echo "Running $model on all datasets for $epochs epochs with a batch size of $batch_size"

        echo "Step 1: Running $model on raw"
        python train.py $model raw $epochs $batch_size

        echo "Step 2: Running $model on morph"
        python train.py $model morph $epochs $batch_size

        echo "Step 3: Running $model on otsu"
        python train.py $model otsu $epochs $batch_size

    elif [ $dataset != "all" ]; then
        echo "Running $model on $dataset for $epochs epochs with a batch size of $batch_size"

        echo "Step 1: Running $model on $dataset"
        python train.py $model $dataset $epochs $batch_size

    fi
    
fi

     
