#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }

while getopts s:t:n:l:p:m:h flag
do
    case "${flag}" in
        s) # Specify scenario used for data processing (1/2/3)
          scenario=${OPTARG};;
        t) # Specify the target variable used for training and prediction ("H_LeThuy", "H_DongHoi", "H_KienGiang")
          target=${OPTARG};;
        n) # Specify the number of time lags for the features
          n_steps=${OPTARG};;
        l) # Specify the number of time leads for the target variable
          lead_time=${OPTARG}
        ;;
        h) # Display help
        usage
        exit 0
        ;;
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

echo "scenario: $scenario"
echo "target: $target"
echo "lag_time: $n_steps"
echo "lead_time: $lead_time"
echo "scenario: $scenario"

python ${PWD}/src/preprocess/preprocess.py --s $scenario --t $target --n $n_steps --l $lead_time \
        > logs/preprocess_`date '+%s'` 2>&1
