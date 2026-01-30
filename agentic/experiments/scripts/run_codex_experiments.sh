#!/usr/bin/env bash
#


set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXPERIMENTS_DIR"

echo "[codex] analysis"
sbatch --wait scripts/analysis-runner.sh

echo "[codex] extract + aggregate"
./scripts/run_extract_and_aggregate_all.sh "$@"

echo "done → ${EXPERIMENTS_DIR}/outputs_extracted/"
