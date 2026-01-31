#!/bin/bash

set -euo pipefail

# run toy analysis
echo "[toy] performing analysis..."
sbatch scripts/toy_old.sh

echo "[toy] extracting + aggregating..."
sbatch scripts/run_extract_and_aggregate_toy.sh

echo "[toy] experiment complete!"