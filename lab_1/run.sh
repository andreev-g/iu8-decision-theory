#!/bin/bash

PROJ_ROOT="$(dirname "$0")/.."
INPUT_FILE="$(dirname "$0")/input/var_3.yaml"
if [ -z "$1" ];
then INPUT_FILE="lab_1/input/var_3.yaml";
else INPUT_FILE="$1"; fi

( \
  cd "$PROJ_ROOT" || return; \
  . venv/bin/activate; \
  python -m lab_1.main "$INPUT_FILE" \
)
