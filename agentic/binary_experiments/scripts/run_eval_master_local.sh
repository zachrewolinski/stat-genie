#!/bin/bash
# Local version (no SLURM). Run from agentic/experiments/.
#
# Usage:
#   cd agentic/experiments
#   bash scripts/run_eval_master_local.sh
#
# For Azure OpenAI: ensure you're logged in with `az login` first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the token refresh helper
source "$SCRIPT_DIR/token-refresh-helper.sh"

eval_script="scripts/run_pairwise_eval.sh"

datasets=("affairs" "amtl" "boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")

# Initial token refresh
refresh_azure_token_if_needed

for dataset in "${datasets[@]}"; do
    # Refresh token if needed (checks if stale)
    refresh_azure_token_if_needed
    
    echo "Running eval for dataset: $dataset"
    LLM_PROVIDER=azureopenai LLM_MODEL=gpt-5-mini-fxdata-eastus2 \
        bash $eval_script $dataset
done
