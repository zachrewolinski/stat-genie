#!/usr/bin/env bash

# Test for the Claude Code extraction pipeline.
# Runs the per-run extractor on each run directory, then aggregates everything
# into a single BLADE-style output directory.
#
#
# Usage:
#   chmod +x agentic/experiments/scripts/test_extract_and_aggregate.sh
#   ./agentic/experiments/scripts/test_extract_and_aggregate.sh

set -euo pipefail

# Run from agentic/experiments (parent of scripts/) so paths match analysis.sh
EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXPERIMENTS_DIR"

# Test target
DATASET="affairs"
PERTURBATION="anonymize"
INPUT_DIR="outputs/${DATASET}/${PERTURBATION}"

# LLM config used for extraction (must exist in config/llm_config.yml)
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"

echo
echo "[1/2] Extracting per-run artifacts from: ${INPUT_DIR}"

for RUN_DIR in "${INPUT_DIR}"/run*; do
  if [ -d "${RUN_DIR}" ]; then
    echo "  - extracting: ${RUN_DIR}"
    poetry run python scripts/extract_single_run.py \
      --run-dir "${RUN_DIR}" \
      --llm-provider "${LLM_PROVIDER}" \
      --llm-model "${LLM_MODEL}"
  fi
done

echo
echo "[2/2] Aggregating into a BLADE-style output directory"

poetry run python scripts/aggregate_runs_to_blade_dir.py \
  --input-dir "${INPUT_DIR}" \
  --overwrite

echo
echo "Done. Aggregated outputs:"
echo "  ${EXPERIMENTS_DIR}/outputs_extracted/${DATASET}/${PERTURBATION}_output/"

