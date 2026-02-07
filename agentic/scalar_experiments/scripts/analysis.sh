#!/bin/bash
# Run from agentic/scalar_experiments/ (job cwd must be scalar_experiments/).

# the first three inputs are dataset name, perturbation type, and run number.
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

# run codex to generate an answer to the research question
poetry run npx codex exec "Follow the instructions given in 'AGENTS.md'" --sandbox workspace-write
