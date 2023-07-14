#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

while getopts s:t:n:l:p:m flag
do
    case "${flag}" in
        s) scenario=${OPTARG};;
        t) target=${OPTARG};;
        n) n_steps=${OPTARG};;
        l) lead_time=${OPTARG};;
        p) pretrain=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

if [[ -z "$scenario" ]]; then
    scenario=1     # 1, 2, or 3
fi

if [[ -z "$target" ]]; then
    target="H_LeThuy"   # if True then use pretrained model, else train new model
fi

if [[ -z "$n_steps" ]]; then
    n_steps=3   # if True then use pretrained model, else train new model
fi

if [[ -z "$lead_time" ]]; then
    lead_time=1   # if True then use pretrained model, else train new model
fi

if [[ -z "$pretrain" ]]; then
    pretrain=True   # if True then use pretrained model, else train new model
fi

if [[ -z "$model" ]]; then
    model=None   # if True then use pretrained model, else train new model
fi

echo "scenario: $scenario"
echo "target: $target"
echo "lag_time: $n_steps"
echo "lead_time: $lead_time"
echo "scenario: $scenario"
echo "pretrain: $pretrain"
echo "model: $model"

python ${PWD}/src/preprocess/preprocess.py --s $scenario --t $target --n $n_steps --l $lead_time --p $pretrain --m $model \
        > logs/preprocess_`date '+%s'` 2>&1
