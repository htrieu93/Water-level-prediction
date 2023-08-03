#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }

while getopts s:n:l:t:m:h flag
do
    case "${flag}" in
        s) # The number of time lags in inputs (same parameter as used in preprocess.sh)
        scenario=${OPTARG}
        ;;
        n) # The number of time lags in inputs (same parameter as used in preprocess.sh)
        n_steps=${OPTARG}
        ;;
        l) # The number of time leads in target column (same parameter as used in preprocess.sh)
        lead_time=${OPTARG}
        ;;
        t) # Target column used for training models
        target=${OPTARG}
        ;;
        m) # Specify the type of models to use (LSTM/Bi-LSTM/GRU)
        model=${OPTARG}
        ;;
        h) # Display help
        usage
        exit 0
        ;;
    esac
done

if [[ -z "$scenario" ]]; then
    scenario=1   # Use scenario 1, 2, or 3
fi

if [[ -z "$n_steps" ]]; then
    n_steps=3   # if True then use pretrained model, else modelling new model
fi

if [[ -z "$lead_time" ]]; then
    lead_time=1   # if True then use pretrained model, else modelling new model
fi

if [[ -z "$target" ]]; then
    target="H_LeThuy"   # if True then use pretrained model, else modelling new model
fi

if [[ -z "$model" ]]; then
    model='LSTM'   # if True then use pretrained model, else modelling new model
elif ! [[ $model =~ ^(LSTM|Bi-LSTM|GRU)$ ]]; then
    echo "Model needs to be LSTM/Bi-LSTM/GRU, $model found instead"
    exit 0
fi

echo "scenario: $scenario"
echo "lag_time: $n_steps"
echo "lead_time: $lead_time"
echo "target: $target"
echo "model: $model"

python ${PWD}/src/utils/set_global_variables.py --s $scenario --n $n_steps \
        --l $lead_time --t $target --m $model \
        > logs/set_global_variables_`date '+%s'` 2>&1
