#!/bin/bash
#
# Run codex analysis, then extract+aggregate. Run from agentic/experiments/.

set -euo pipefail

echo "[codex] analysis"
sbatch --wait scripts/analysis-runner.sh

echo "[gpt-5-mini] extract + aggregate"
sbatch --wait scripts/run_extract_and_aggregate_all.sh

echo "done -> outputs_extracted/"

echo "[gpt-5-mini] pairwise judge"
sbatch scripts/run_eval_master.sh