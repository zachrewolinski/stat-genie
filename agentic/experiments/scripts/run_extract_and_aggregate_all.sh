#!/usr/bin/env bash
#
# Extract every run under outputs/<dataset>/<perturbation>/, then aggregate
# into outputs_extracted/<dataset>/<perturbation>_output/.
# Run from agentic/experiments/. Requires analysis.py, info.json, conclusion.txt in each run dir.
# Skips runs that already have extracted_analysis.json. Use --overwrite to re-extract.

set -euo pipefail

OUTPUTS_ROOT="outputs"
EXTRACTED_ROOT="outputs_extracted"

LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"

OVERWRITE=""
for arg in "$@"; do
  if [[ "$arg" == "--overwrite" ]]; then
    OVERWRITE=1
    break
  fi
done

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
    echo "[${dataset}/${perturbation}] ${run_count} run(s)"
    for run_dir in "${pert_dir}"run*; do
      [[ -d "$run_dir" ]] || continue
      if [[ -n "$OVERWRITE" ]] || [[ ! -f "${run_dir}/extracted_analysis.json" ]]; then
        echo "  - ${run_dir}"
        poetry run python scripts/extract_single_run.py \
          --run-dir "$run_dir" \
          --llm-provider "$LLM_PROVIDER" \
          --llm-model "$LLM_MODEL"
      else
        echo "  - ${run_dir} (skip)"
      fi
    done

    echo "[${dataset}/${perturbation}] aggregate"
    poetry run python scripts/aggregate_runs_to_blade_dir.py \
      --input-dir "$pert_dir" \
      --overwrite
  done
done

echo
echo "done -> ${EXTRACTED_ROOT}/"
