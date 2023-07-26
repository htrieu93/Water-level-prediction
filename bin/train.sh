#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

while getopts s:t:n:l:p:m flag
do
    case "${flag}" in
        p) pretrain=${OPTARG};;
        m) model=${OPTARG};;
        n) n_steps=${OPTARG};;
        l) lead_time=${OPTARG};;
    esac
done

if [[ -z "$model" ]]; then
    model='LSTM'   # if True then use pretrained model, else train new model
fi
if [[ -z "$n_steps" ]]; then
    n_steps=3   # if True then use pretrained model, else train new model
fi

if [[ -z "$lead_time" ]]; then
    lead_time=1   # if True then use pretrained model, else train new model
fi

echo "pretrain: $pretrain"
echo "model: $model"
echo "lag_time: $n_steps"
echo "lead_time: $lead_time"

python ${PWD}/src/train/train.py --p $pretrain --m $model --n $n_steps --l $lead_time \
        > logs/train_`date '+%s'` 2>&1
