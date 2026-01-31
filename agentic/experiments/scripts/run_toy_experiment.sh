#!/usr/bin/env bash
#
# Run toy analysis, then extract+aggregate.

set -euo pipefail

echo "[toy] analysis"
sbatch --wait scripts/toy.sh

echo "[toy] extracting + aggregating"
sbatch scripts/run_extract_and_aggregate_toy.sh
