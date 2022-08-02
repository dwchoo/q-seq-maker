#!/bin/bash

activate () {
    SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
    source $SCRIPTPATH/.venv/bin/activate
}

activate

python main.py \
    --name Data_011 \
    --num_set 10 \
    --threshold 7 \
    --PAM NGG \
    --path ./data \
    --method generate_8_nC2_data \
    -v
