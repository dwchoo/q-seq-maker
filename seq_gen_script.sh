#!/bin/bash

activate () {
    SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

    python -m pip install -U pip
    python -m pip install virtualenv
    python -m venv .venv --withou-pip
    curl https://bootstrap.pypa.io/get-pip.py | $SCRIPTPATH/.venv/bin/python

    source $SCRIPTPATH/.venv/bin/activate
    python -m pip install -r $SCRIPTPATH/requirements.txt
}

activate

python main.py \
    --name Data_011 \
    --num_set 10000000 \
    --threshold 7 \
    --PAM NGG \
    --path ./data \
    --method generate_8_nC2_data \
    -v
