#!/bin/bash

# the two inputs are dataset name and perturbation type.
dataset_name="$1"
perturbation_type="$2"
run_number="$3"

# call make-subdir.py with the given dataset name and perturbation type
poetry run python scripts/make-subdir.py \
    --dataset "$dataset_name" \
    --perturbation-type "$perturbation_type" \
    --run_number "$run_number"

# change directory to the newly created subdirectory
cd "outputs/$dataset_name/$perturbation_type/run$run_number" || exit 1

# disable use of cache for claude code
export DISABLE_PROMPT_CACHING=1

# run claude code to generate an answer to the research question
poetry run claude -p "Follow the instructions given in 'instructions.txt'" --allowedTools "Read,Edit,Bash" --dangerously-skip-permissions