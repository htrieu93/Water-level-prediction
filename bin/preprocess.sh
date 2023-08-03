#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"

python ${PWD}/src/modelling/preprocess.py \
        > logs/preprocess_`date '+%s'` 2>&1
