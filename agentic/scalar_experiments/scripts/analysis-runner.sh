#!/bin/bash
# Run from agentic/scalar_experiments/ (or sbatch from there so job cwd is scalar_experiments/).

# list all blade datasets
# datasets=("affairs" "amtl" "boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")
datasets=("affairs" "amtl" "boxes") # use first three for testing purposes

# list all perturbation types
# perturbations=("null_anonymize" "null_shuffle_names" "null_add_features" "null_positive_leading_statement" "null_negative_leading_statement")
perturbations=("null_anonymize" "null_shuffle_names") # use first two for testing purposes

# number of runs per dataset-perturbation pair
# num_runs=20
num_runs=5 # use 5 for testing purposes

# analysis script name
analysis_script="scripts/analysis.sh"

# for each dataset-perturbation pair, run analysis.sh five times
for dataset in "${datasets[@]}"; do
    for perturbation in "${perturbations[@]}"; do
        for run_number in $(seq 1 $num_runs); do
            echo "[analysis-runner] Running analysis for dataset: $dataset, perturbation: $perturbation, run number: $run_number"
            sbatch --wait $analysis_script $dataset $perturbation $run_number
        done
    done
done