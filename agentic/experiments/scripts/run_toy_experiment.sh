#!/usr/bin/env bash
#
# Run toy analysis (toy.sh), then extract and aggregate into toy_extracted/.
# Requires analysis.py, info.json, conclusion.txt in toy/ after analysis.

set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOY_DIR="${EXPERIMENTS_DIR}/toy"
TOY_RUNS="${EXPERIMENTS_DIR}/toy_runs"
TOY_EXTRACTED="${EXPERIMENTS_DIR}/toy_extracted"

LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"

cd "$EXPERIMENTS_DIR"

echo "[toy] analysis"
./scripts/toy.sh

echo "[toy] extract"
poetry run python scripts/extract_single_run.py \
  --run-dir "$TOY_DIR" \
  --llm-provider "$LLM_PROVIDER" \
  --llm-model "$LLM_MODEL"

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
