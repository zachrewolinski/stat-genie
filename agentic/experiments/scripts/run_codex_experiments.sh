#!/usr/bin/env bash
#
# Run codex analysis, then extract+aggregate. Run from agentic/experiments/.

set -euo pipefail

echo "[codex] analysis"
sbatch --wait scripts/analysis-runner.sh

echo "[codex] extract + aggregate"
./scripts/run_extract_and_aggregate_all.sh "$@"

echo "done -> outputs_extracted/"
