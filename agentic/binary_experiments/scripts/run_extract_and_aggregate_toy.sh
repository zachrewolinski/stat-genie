#!/bin/bash
#
# Extract from toy/run*, aggregate into toy_extracted/. Overwrites by default.

set -euo pipefail

TOY_DIR="toy"
TOY_EXTRACTED="toy_extracted"

LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"

echo "working directory:"
pwd

echo "[toy] extract"

poetry run python scripts/extract_single_run.py \
    --run-dir "toy/run1" \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL"

echo "[toy] aggregate"
poetry run python scripts/aggregate_runs_to_blade_dir.py \
  --input-dir "$TOY_DIR" \
  --output-dir "$TOY_EXTRACTED" \
  --overwrite

echo
echo "done → ${TOY_EXTRACTED}"
