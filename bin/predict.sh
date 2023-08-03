#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }

while getopts p:h flag
do
    case "${flag}" in
        p) # Whether to use pretrained model or not
          pretrain=${OPTARG}
        ;;
        h) # Display help
        usage
        exit 0
        ;;
    esac
done

python ${PWD}/src/modelling/predict.py \
        > logs/predict_`date '+%s'` 2>&1
