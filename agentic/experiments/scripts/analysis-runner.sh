#!/bin/bash

# list all blade datasets
# datasets=("affairs" "amtl" "boxes" "caschools" "crofoot" "fish" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")
datasets=("amtl")

# list all perturbation types
# perturbations=("noperturb" "anonymize" "shuffle_names" "add_features" "replace_with_rvs" "positive_leading_statement" "negative_leading_statement" "replace_and_positive_statement")
# perturbations=("noperturb" "anonymize" "replace_with_rvs" "positive_leading_statement")
perturbations=("shuffle_names")

# number of runs per dataset-perturbation pair
num_runs=2

# analysis script name
analysis_script="scripts/analysis.sh"

# agent to use
agent_name="codex"

# for each dataset-perturbation pair, run analysis.sh five times
for dataset in "${datasets[@]}"; do
    for perturbation in "${perturbations[@]}"; do
        for run_number in $(seq 1 $num_runs); do
            echo "[analysis-runner] Running analysis for dataset: $dataset, perturbation: $perturbation, run number: $run_number"
            sbatch --wait $analysis_script $dataset $perturbation $run_number $agent_name
        done
    done
done