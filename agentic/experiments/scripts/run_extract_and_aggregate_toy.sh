#!/usr/bin/env bash
#
# Extract from agentic/experiments/toy/ (single run), then aggregate into
# agentic/experiments/toy_extracted/. Requires analysis.py, info.json, conclusion.txt
# in toy/ (produced by scripts/toy.sh).
#
# Skips extraction if toy/extracted_analysis.json exists (use --overwrite to re-extract).
# Uses toy_runs/run1 -> ../toy so the aggregator sees one run dir.

set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOY_DIR="${EXPERIMENTS_DIR}/toy"
TOY_RUNS="${EXPERIMENTS_DIR}/toy_runs"
TOY_EXTRACTED="${EXPERIMENTS_DIR}/toy_extracted"

LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"

OVERWRITE=""
for arg in "$@"; do
  if [[ "$arg" == "--overwrite" ]]; then
    OVERWRITE=1
    break
  fi
done

cd "$EXPERIMENTS_DIR"

echo
echo "[toy] extract"
if [[ -n "$OVERWRITE" ]] || [[ ! -f "${TOY_DIR}/extracted_analysis.json" ]]; then
  echo "  - ${TOY_DIR}"
  poetry run python scripts/extract_single_run.py \
    --run-dir "$TOY_DIR" \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL"
else
  echo "  - ${TOY_DIR} (skip)"
fi

echo "[toy] aggregate"
mkdir -p "$TOY_RUNS"
rm -f "$TOY_RUNS/run1"
ln -s ../toy "$TOY_RUNS/run1"

poetry run python scripts/aggregate_runs_to_blade_dir.py \
  --input-dir "$TOY_RUNS" \
  --output-dir "$TOY_EXTRACTED" \
  --overwrite

echo
echo "done → ${TOY_EXTRACTED}"
