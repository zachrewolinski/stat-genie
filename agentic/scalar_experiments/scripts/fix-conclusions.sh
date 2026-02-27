#!/bin/bash
# Fix invalid JSON in conclusion.txt files.
# Run from agentic/scalar_experiments/.
#
# Usage:
#   cd agentic/scalar_experiments
#   bash scripts/fix-conclusions.sh [--dry-run] [--verbose]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Only use Azure token refresh if OPENAI_API_KEY isn't already set.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    source "$SCRIPT_DIR/token-refresh-helper.sh"
    if ! refresh_azure_token_until_ok; then
        echo "[fix-conclusions] WARNING: No OPENAI_API_KEY and Azure token refresh failed." >&2
    fi
fi

cd "$EXP_ROOT"
poetry run python scripts/fix_conclusions.py "$@"
