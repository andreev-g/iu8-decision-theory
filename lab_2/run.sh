#!/bin/bash

PROJ_ROOT="$(dirname "$0")/.."
if [ -z "$1" ];
then INPUT_FILE="lab_2/input/var_3.yaml";
else INPUT_FILE="$1"; fi

( \
  cd "$PROJ_ROOT" || return; \
  . venv/bin/activate; \
  python -m lab_2.main "$INPUT_FILE" \
)
