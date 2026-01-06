#!/bin/bash

TOP_DIR=$(pwd)
DATA_FOLDER="${TOP_DIR}/data"
MODEL_FOLDER="${TOP_DIR}/pre-trained-models"

##### 1. Generate datasets #####
# Input: CSV file containing SMILES and target values
# Output: train, val, test data loaders saved in defined folder

python $TOP_DIR/r2s2_dataset.py \
    --smiles-path $DATA_FOLDER/CombiSolv-QM-sample-1.csv \
    --save-path $DATA_FOLDER
    # --xyz-aqm-path $DATA_FOLDER/AQM-sample-1.csv \

##### 2. Train the model #####
# Input: train, val, test data loaders
# Output: model saved in defined folder

python $TOP_DIR/r2s2_train.py \
    --train-set $DATA_FOLDER/train_loader.pth \
    --val-set $DATA_FOLDER/val_loader.pth \
    --test-set $DATA_FOLDER/test_loader.pth \
    --epochs 3 \
    --save-model $MODEL_FOLDER \
    --device cpu

##### 3. Predict using the pretrained model #####
# Input: pretrained model and test data loader
# Output: prediction results

python $TOP_DIR/r2s2_predict.py \
    --model $MODEL_FOLDER/r2s2_model_epoch1.pth \
    --test-set $DATA_FOLDER/test_loader.pth \
    --device cpu

##### 4. Cleaning #####

rm $DATA_FOLDER/*.pth
rm $MODEL_FOLDER/r2s2_model_epoch*.pth
