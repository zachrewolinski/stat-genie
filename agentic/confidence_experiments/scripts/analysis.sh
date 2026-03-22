#!/bin/bash

# run from agentic/scalar_experiments/ (job cwd must be scalar_experiments/).

# the first four inputs are prompt version, dataset name, perturbation type, and run number.
# prompt_version="$1"
dataset_name="$1"
distribution="$2"
perturbation_type="$3"
run_number="$4"
# there is a fifth argument that is only used if the distribution is "pve", which is the pve value to use for that run's dataset perturbation
if [[ "$distribution" == "pve" ]]; then
    pve="$5"
fi

# call make-subdir.py with the given dataset name and perturbation type
if [[ "$distribution" == "pve" ]]; then
    poetry run python scripts/make-subdir.py \
        --dataset "$dataset_name" \
        --distribution "$distribution" \
        --perturbation-type "$perturbation_type" \
        --run_number "$run_number" \
        --pve "$pve"
else
    poetry run python scripts/make-subdir.py \
        --dataset "$dataset_name" \
        --distribution "$distribution" \
        --perturbation-type "$perturbation_type" \
        --run_number "$run_number"
fi

# change directory to the newly created subdirectory
if [[ "$distribution" == "pve" ]]; then
    cd "outputs/$dataset_name/$distribution/pve_$pve/$perturbation_type/run$run_number" || exit 1
else
    cd "outputs/$dataset_name/$distribution/$perturbation_type/run$run_number" || exit 1
fi

# run codex to generate an answer to the research question
poetry run npx codex exec --config model_reasoning_effort="high" --sandbox workspace-write "Follow the instructions given in 'AGENTS.md'" 
