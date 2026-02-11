#!/bin/bash
# Local version (no SLURM). Run from agentic/scalar_experiments/.
#
# Usage:
#   cd agentic/scalar_experiments
#   bash scripts/analysis-runner-local.sh
#
# For Azure OpenAI: ensure you're logged in with `az login` first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the token refresh helper
source "$SCRIPT_DIR/token-refresh-helper.sh"

# define three test datasets for inspecting prompts
datasets=("affairs" "amtl" "caschools")

# list all perturbation types
perturbations=("null_anonymize" "null_shuffle_names" "null_add_features" "null_positive_leading_statement" "null_negative_leading_statement")

# 10 runs to speed things up
num_runs=10

# prompt version (1-4)
prompt_versions=(1 2 3 4)

# analysis script name
analysis_script="scripts/analysis.sh"

# Initial token refresh
refresh_azure_token_if_needed

# for each dataset-perturbation pair, run analysis.sh `num_runs` times
for prompt_version in "${prompt_versions[@]}"; do
    for dataset in "${datasets[@]}"; do
        for perturbation in "${perturbations[@]}"; do
            for run_number in $(seq 1 $num_runs); do
                # Check if this experiment already has output (conclusion.txt indicates completion)
                output_dir="outputs/prompt$prompt_version/$dataset/$perturbation/run$run_number"
                if [ -f "$output_dir/conclusion.txt" ]; then
                    echo "[analysis-runner] Skipping (already completed): $dataset, $perturbation, run $run_number"
                    continue
                fi
                
                # Refresh token if needed (checks if stale)
                refresh_azure_token_if_needed
                
                echo "[analysis-runner] Running analysis for dataset: $dataset, perturbation: $perturbation, run number: $run_number"
                bash $analysis_script $prompt_version $dataset $perturbation $run_number
            done
        done
    done
done
