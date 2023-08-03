#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

python ${PWD}/src/modelling/train.py \
        > logs/train_`date '+%s'` 2>&1
