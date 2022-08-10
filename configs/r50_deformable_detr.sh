#!/usr/bin/env bash

set -x

PY_ARGS=${@:1}

python -u main.py \
    ${PY_ARGS}
