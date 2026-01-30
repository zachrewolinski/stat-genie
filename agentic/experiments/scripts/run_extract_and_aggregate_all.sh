#!/usr/bin/env bash
#
# Run per-run extraction for every run in every dataset/perturbation under
# agentic/experiments/outputs/, then aggregate each (dataset, perturbation)
# into agentic/experiments/outputs_extracted/<dataset>/<perturbation>_output/.
#
# Prerequisites: analysis.py, info.json, conclusion.txt in each run directory.
#
# Usage: run from anywhere (script cds to agentic/experiments internally).
#   chmod +x agentic/experiments/scripts/run_extract_and_aggregate_all.sh
#   ./agentic/experiments/scripts/run_extract_and_aggregate_all.sh
# Or from agentic/experiments: ./scripts/run_extract_and_aggregate_all.sh

set -euo pipefail

# Run from agentic/experiments (parent of scripts/) so paths match analysis.sh / analysis-runner.sh
EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS_ROOT="outputs"
EXTRACTED_ROOT="outputs_extracted"

LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"

cd "$EXPERIMENTS_DIR"

for dataset_dir in "${OUTPUTS_ROOT}"/*/; do
  [[ -d "$dataset_dir" ]] || continue
  dataset=$(basename "$dataset_dir")
  for pert_dir in "${dataset_dir}"*/; do
    [[ -d "$pert_dir" ]] || continue
    perturbation=$(basename "$pert_dir")

    run_count=0
    for run_dir in "${pert_dir}"run*; do
      [[ -d "$run_dir" ]] || continue
      ((run_count++)) || true
    done
    [[ "$run_count" -gt 0 ]] || continue

    echo
    echo "[${dataset}/${perturbation}] Extracting ${run_count} run(s)..."
    for run_dir in "${pert_dir}"run*; do
      [[ -d "$run_dir" ]] || continue
      echo "  - ${run_dir}"
      poetry run python scripts/extract_single_run.py \
        --run-dir "$run_dir" \
        --llm-provider "$LLM_PROVIDER" \
        --llm-model "$LLM_MODEL"
    done

    echo "[${dataset}/${perturbation}] Aggregating..."
    poetry run python scripts/aggregate_runs_to_blade_dir.py \
      --input-dir "$pert_dir" \
      --overwrite
  done
done

echo
echo "Done. Aggregated outputs under: ${EXPERIMENTS_DIR}/${EXTRACTED_ROOT}/"
