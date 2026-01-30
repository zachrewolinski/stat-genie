#!/usr/bin/env bash
#
# Extract from toy/run*, aggregate into toy_extracted/. Overwrites by default.

set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOY_DIR="${EXPERIMENTS_DIR}/toy"
TOY_EXTRACTED="${EXPERIMENTS_DIR}/toy_extracted"

LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"

cd "$EXPERIMENTS_DIR"

echo
echo "[toy] extract"
for run_dir in "${TOY_DIR}"/run*; do
  [[ -d "$run_dir" ]] || continue
  echo "  - ${run_dir}"
  poetry run python scripts/extract_single_run.py \
    --run-dir "$run_dir" \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL"
done

echo "[toy] aggregate"
poetry run python scripts/aggregate_runs_to_blade_dir.py \
  --input-dir "$TOY_DIR" \
  --output-dir "$TOY_EXTRACTED" \
  --overwrite

echo
echo "done → ${TOY_EXTRACTED}"
