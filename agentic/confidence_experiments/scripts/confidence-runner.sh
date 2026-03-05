#!/bin/bash
#SBATCH --job-name=agent-confidence-runner
#SBATCH --output=slurm_output/%x-%j.out

# run from agentic/scalar_experiments/ (or sbatch from there so job cwd is scalar_experiments/).

# list all blade datasets
# datasets=("affairs" "amtl" "boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")
# have already done "affairs" and "amtl" so skipping those for now
# datasets=("boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")
datasets=("affairs" "amtl")

# list the distribution types
distributions=("alt" "null")

# get pve values
# pves=(0.0 0.5 1.0) # only used if distribution is "pve"

# list all perturbation types
perturbations=("anonymize") # "shuffle_names" "add_features" "positive_leading_statement" "negative_leading_statement")

# number of runs per dataset-perturbation pair
# num_runs=20
num_runs=1

# prompt version (1-4)
# prompt_versions=(1 2 3 4)

# analysis script name
analysis_script="scripts/confidence.sh"

# for each dataset-perturbation pair, run analysis.sh `num_runs` times
for distribution in "${distributions[@]}"; do
    if [[ "$distribution" == "pve" ]]; then
        pve_list=("${pves[@]}")
    else
        pve_list=("")
    fi
    for dataset in "${datasets[@]}"; do
        for perturbation in "${perturbations[@]}"; do
            for pve in "${pve_list[@]}"; do
                for run_number in $(seq 1 $num_runs); do
                    if [[ "$distribution" == "pve" ]]; then
                        echo "[confidence-runner] Running analysis for dataset: $dataset, distribution: $distribution, pve: $pve, perturbation: $perturbation, run number: $run_number"
                        sbatch --wait $analysis_script $dataset $distribution $perturbation $run_number $pve
                    else
                        echo "[confidence-runner] Running analysis for dataset: $dataset, distribution: $distribution, perturbation: $perturbation, run number: $run_number"
                        sbatch --wait $analysis_script $dataset $distribution $perturbation $run_number
                    fi
                done
            done
        done
    done
done