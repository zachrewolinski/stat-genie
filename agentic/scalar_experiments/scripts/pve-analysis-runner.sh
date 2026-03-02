#!/bin/bash
#SBATCH --job-name=agent-runner
#SBATCH --output=slurm_output/%x-%j.out

# run from agentic/scalar_experiments/ (or sbatch from there so job cwd is scalar_experiments/).

# we want to run this mini-experiment on the 'amtl' dataset since it has lots
# of signal in the alternative distribution
datasets=("amtl")

# only interested in varying pve
distributions=("pve")

# get pve values
pves=(0.0 0.2 0.4 0.6) # only used if distribution is "pve"

# list all perturbation types
perturbations=("anonymize" "shuffle_names" "add_features" "positive_leading_statement" "negative_leading_statement")

# five runs per perturbation should get us 100 total runs per dataset
num_runs=5

# analysis script name
analysis_script="scripts/analysis.sh"

# for each dataset-perturbation pair, run analysis.sh `num_runs` times
for distribution in "${distributions[@]}"; do
    if [[ "$distribution" == "pve" ]]; then
        pve_list=("${pves[@]}")
    else
        # pve_list=("")
        # throw error, since this is the pve runner
        echo "[ERROR] 'pve-analysis-runner.sh' only accepts 'pve' as the distribution type."
        exit 1
    fi
    for dataset in "${datasets[@]}"; do
        for perturbation in "${perturbations[@]}"; do
            for pve in "${pve_list[@]}"; do
                for run_number in $(seq 1 $num_runs); do
                    if [[ "$distribution" == "pve" ]]; then
                        echo "[analysis-runner] Running analysis for dataset: $dataset, distribution: $distribution, pve: $pve, perturbation: $perturbation, run number: $run_number"
                        sbatch --wait $analysis_script $dataset $distribution $perturbation $run_number $pve
                    else
                        # echo "[analysis-runner] Running analysis for dataset: $dataset, distribution: $distribution, perturbation: $perturbation, run number: $run_number"
                        # sbatch --wait $analysis_script $dataset $distribution $perturbation $run_number
                        # pve_list=("")
                        # throw error, since this is the pve runner
                        echo "[ERROR] 'pve-analysis-runner.sh' only accepts 'pve' as the distribution type."
                        exit 1
                    fi
                done
            done
        done
    done
done