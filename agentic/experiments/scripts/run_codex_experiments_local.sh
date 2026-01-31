#!/bin/bash
#
# Run codex analysis, then extract+aggregate (local version, no SLURM).
# Run from agentic/experiments/.
#
# Usage:
#   cd agentic/experiments
#   bash scripts/run_codex_experiments_local.sh
#
# For Azure OpenAI: ensure you're logged in with `az login` first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the token refresh helper
source "$SCRIPT_DIR/token-refresh-helper.sh"

# Initial token refresh for Azure
refresh_azure_token_if_needed

echo "[codex] analysis"
bash "$SCRIPT_DIR/analysis-runner-local.sh"

echo "[gpt-5-mini] extract + aggregate"
refresh_azure_token_if_needed  # Refresh before extraction in case token expired
bash "$SCRIPT_DIR/run_extract_and_aggregate_all.sh"

echo "done -> outputs_extracted/"

echo "[gpt-5-mini] pairwise judge"
refresh_azure_token_if_needed  # Refresh before eval in case token expired
bash "$SCRIPT_DIR/run_eval_master_local.sh"

echo "All steps completed!"
