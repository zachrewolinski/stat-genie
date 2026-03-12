#!/bin/bash
# Local version (no SLURM). Run from agentic/scalar_experiments/.
#
# Usage:
#   cd agentic/scalar_experiments
#   bash scripts/pve-runner-local.sh
#
# For Azure OpenAI: ensure you're logged in with `az login` first.
# Uses retries for token refresh and for each analysis run (stream/API failures).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Experiment root (scalar_experiments) so paths work regardless of cwd
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source the token refresh helper
source "$SCRIPT_DIR/token-refresh-helper.sh"

# put all datasets for full suite of results
datasets=("affairs" "amtl" "boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")

# only interested in varying pve
distributions=("pve")

# get pve values
pves=(0.0 0.001 0.005 0.01 0.05 0.2)

# list all perturbation types
perturbations=("anonymize" "shuffle_names" "add_features" "positive_leading_statement" "negative_leading_statement")

# 20 runs each for the full distributions
num_runs=20

# analysis script name
analysis_script="scripts/analysis.sh"

# Retry config for analysis runs (stream disconnects, response.failed, etc.)
MAX_ANALYSIS_ATTEMPTS="${MAX_ANALYSIS_ATTEMPTS:-3}"
ANALYSIS_RETRY_DELAY="${ANALYSIS_RETRY_DELAY:-25}"

# Initial token refresh with retries (exit if we can't get a token at all)
if ! refresh_azure_token_until_ok; then
    echo "[analysis-runner] FATAL: Could not obtain Azure token after retries. Check \`az login\` and network." >&2
    exit 1
fi

# for each dataset-perturbation pair, run analysis.sh `num_runs` times
for distribution in "${distributions[@]}"; do
    if [[ "$distribution" == "pve" ]]; then
        pve_list=("${pves[@]}")
    else
        # throw error, since this is the pve runner
        echo "[ERROR] 'pve-runner-local.sh' only accepts 'pve' as the distribution type."
        exit 1
    fi
    for dataset in "${datasets[@]}"; do
        for perturbation in "${perturbations[@]}"; do
            for pve in "${pve_list[@]}"; do
                for run_number in $(seq 1 $num_runs); do
                    # skip if this experiment already has output (conclusion.txt indicates completion)
                    output_dir="$EXP_ROOT/outputs/$dataset/$distribution/pve_$pve/$perturbation/run$run_number"
                    if [ -f "$output_dir/conclusion.txt" ]; then
                        # echo "[analysis-runner] Skipping (already completed): prompt$prompt_version $dataset $perturbation run$run_number"
                        echo "[analysis-runner] Skipping (already completed): $dataset $distribution $pve $perturbation run$run_number"
                        continue
                    fi

                    # Refresh token before each run (with retries) to avoid stale/expired tokens
                    refresh_azure_token_until_ok || true

                    attempt=1
                    while true; do
                        # echo "[analysis-runner] Running analysis (attempt $attempt/$MAX_ANALYSIS_ATTEMPTS): prompt$prompt_version $dataset $perturbation run$run_number"
                        echo "[analysis-runner] Running analysis (attempt $attempt/$MAX_ANALYSIS_ATTEMPTS): $dataset $distribution pve_$pve $perturbation run$run_number"
                        (cd "$EXP_ROOT" && bash $analysis_script $dataset $distribution $perturbation $run_number $pve) || true
                        if [ -f "$output_dir/conclusion.txt" ]; then
                            # echo "[analysis-runner] Completed: prompt$prompt_version $dataset $perturbation run$run_number"
                            echo "[analysis-runner] Completed: $dataset $distribution pve_$pve $perturbation run$run_number"
                            break
                        fi

                        if [ $attempt -ge "$MAX_ANALYSIS_ATTEMPTS" ]; then
                            # echo "[analysis-runner] FAILED after $MAX_ANALYSIS_ATTEMPTS attempts: prompt$prompt_version $dataset $perturbation run$run_number (continuing to next run)" >&2
                            echo "[analysis-runner] FAILED after $MAX_ANALYSIS_ATTEMPTS attempts: $dataset $distribution pve_$pve $perturbation run$run_number (continuing to next run)" >&2
                            break
                        fi

                        echo "[analysis-runner] Run failed; refreshing token and retrying in ${ANALYSIS_RETRY_DELAY}s..."
                        refresh_azure_token_until_ok || true
                        sleep "$ANALYSIS_RETRY_DELAY"
                        ((attempt++)) || true
                    done
                done
            done
        done
    done
done

echo "[analysis-runner] Done."
