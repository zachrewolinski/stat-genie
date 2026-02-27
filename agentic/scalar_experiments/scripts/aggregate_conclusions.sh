#!/bin/bash
# Run from agentic/scalar_experiments/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fix broken conclusions first (programmatic + LLM fallback)
bash "$SCRIPT_DIR/fix-conclusions.sh"

poetry run python scripts/aggregate_conclusions.py
