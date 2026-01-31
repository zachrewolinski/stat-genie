#!/bin/bash
# Local version (no SLURM). Run from agentic/experiments/.
#
# Usage:
#   cd agentic/experiments
#   bash scripts/analysis-runner-local.sh
#
# For Azure OpenAI: ensure you're logged in with `az login` first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the token refresh helper
source "$SCRIPT_DIR/token-refresh-helper.sh"

# list all blade datasets
datasets=("affairs" "amtl" "boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")

# list all perturbation types
perturbations=("noperturb" "anonymize" "shuffle_names" "add_features" "replace_with_rvs" "positive_leading_statement" "negative_leading_statement" "replace_and_positive_statement")

# number of runs per dataset-perturbation pair
num_runs=5

# analysis script name
analysis_script="scripts/analysis.sh"

# agent to use
agent_name="codex"

# Initial token refresh
refresh_azure_token_if_needed

# for each dataset-perturbation pair, run analysis.sh five times
for dataset in "${datasets[@]}"; do
    for perturbation in "${perturbations[@]}"; do
        for run_number in $(seq 1 $num_runs); do
            # Refresh token if needed (checks if stale)
            refresh_azure_token_if_needed
            
            echo "[analysis-runner] Running analysis for dataset: $dataset, perturbation: $perturbation, run number: $run_number"
            bash $analysis_script $dataset $perturbation $run_number $agent_name
        done
    done
done
