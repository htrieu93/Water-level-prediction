#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

while getopts s:t:n:l:p:m flag
do
    case "${flag}" in
        p) pretrain=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

if [[ -z "$pretrain" ]]; then
    pretrain=False   # if True then use pretrained model, else train new model
fi

if [[ -z "$model" ]]; then
    model='LSTM'   # if True then use pretrained model, else train new model
fi

echo "pretrain: $pretrain"
echo "model: $model"

python ${PWD}/src/train/train.py --p $pretrain --m $model \
        > logs/train_`date '+%s'` 2>&1
