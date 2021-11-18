#!/bin/bash

PROJ_ROOT="$(dirname "$0")/.."

( \
  cd "$PROJ_ROOT" || return; \
  . venv/bin/activate; \
  python -m lab_1.main "lab_1/input_data.yaml" "$1" \
)
