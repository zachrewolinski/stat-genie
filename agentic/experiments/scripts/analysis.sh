#!/bin/bash

# the two inputs are dataset name and perturbation type.
dataset_name="$1"
perturbation_type="$2"

# call make-subdir.py with the given dataset name and perturbation type
poetry run python scripts/make-subdir.py \
    --dataset "$dataset_name" \
    --perturbation-type "$perturbation_type"

