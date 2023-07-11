#!/bin/bash

python ${PWD}/src/preprocess/preprocess.py \
        > logs/preprocess_`date '+%s'` 2>&1
