#!/bin/bash
# Fix invalid JSON in conclusion.txt and confidence.txt files.
# Run from agentic/confidence_experiments/.
#
# Usage:
#   cd agentic/confidence_experiments
#   bash scripts/fix-conclusions.sh [--dry-run] [--verbose]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$EXP_ROOT"
poetry run python scripts/fix_conclusions.py "$@"
