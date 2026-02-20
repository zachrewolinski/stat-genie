#!/bin/bash
#SBATCH --job-name=agent-analysis
#SBATCH --output=slurm_output/%x-%j.out

# run from agentic/scalar_experiments/ (job cwd must be scalar_experiments/).

# the first four inputs are prompt version, dataset name, perturbation type, and run number.
# prompt_version="$1"
dataset_name="$1"
distribution="$2"
perturbation_type="$3"
run_number="$4"

# call make-subdir.py with the given dataset name and perturbation type
poetry run python scripts/make-subdir.py \
    --dataset "$dataset_name" \
    --distribution "$distribution" \
    --perturbation-type "$perturbation_type" \
    --run_number "$run_number"

# change directory to the newly created subdirectory
# cd "outputs/prompt$prompt_version/$dataset_name/$perturbation_type/run$run_number" || exit 1
cd "outputs/$dataset_name/$distribution/$perturbation_type/run$run_number" || exit 1

# run codex to generate an answer to the research question
poetry run npx codex exec --config model_reasoning_effort="high" --sandbox workspace-write "Follow the instructions given in 'AGENTS.md'" 
