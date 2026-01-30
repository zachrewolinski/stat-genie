#!/usr/bin/env bash
#
# Run toy analysis (toy.sh), then extract + aggregate via run_extract_and_aggregate_toy.sh.

set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXPERIMENTS_DIR"

echo "[toy] analysis"
./scripts/toy.sh

./scripts/run_extract_and_aggregate_toy.sh "$@"
