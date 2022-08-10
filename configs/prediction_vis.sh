#!/usr/bin/env bash

set -x

PY_ARGS=${@:1}

python -u prediction.py ${PY_ARGS}
