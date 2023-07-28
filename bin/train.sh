#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }

while getopts m:p:n:l:h flag
do
    case "${flag}" in
        m) # Specify the type of models to use (LSTM/Bi-LSTM/GRU)
          model=${OPTARG}
        ;;
        p) # Whether to use pretrained model or not
          pretrain=${OPTARG}
        ;;
        n) # The number of time lags in inputs (same parameter as used in preprocess.sh)
        n_steps=${OPTARG}
        ;;
        l) # The number of time leads in target column (same parameter as used in preprocess.sh)
        lead_time=${OPTARG}
        ;;
        h) # Display help
        usage
        exit 0
        ;;
    esac
done

if ! [[ $model =~ ^(LSTM|Bi-LSTM|GRU)$ ]]; then
    echo "Model needs to be LSTM/Bi-LSTM/GRU, $model found instead"
    exit 0
elif [[ -z "$model" ]]; then
    model='LSTM'   # if True then use pretrained model, else train new model
    echo "Empty string"
fi

if [[ -z "$n_steps" ]]; then
    n_steps=3   # if True then use pretrained model, else train new model
fi

if [[ -z "$lead_time" ]]; then
    lead_time=1   # if True then use pretrained model, else train new model
fi

echo "model: $model"
echo "lag_time: $n_steps"
echo "lead_time: $lead_time"

python ${PWD}/src/train/train.py --m $model --n $n_steps --l $lead_time \
        > logs/train_`date '+%s'` 2>&1
