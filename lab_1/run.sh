#!/bin/bash

. venv/bin/activate

python -m lab_1.src.main "lab_1/input_data.yaml" "$1"
