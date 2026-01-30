#!/bin/bash
#
# Create toy/run1/, copy in instructions and data, run codex there. Outputs go into toy/run1/.

set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOY_DIR="${EXPERIMENTS_DIR}/toy"
RUN_DIR="${TOY_DIR}/run1"

cd "$EXPERIMENTS_DIR"
mkdir -p "$RUN_DIR"
cp "$TOY_DIR"/AGENTS.md "$TOY_DIR"/info.json "$TOY_DIR"/compas.csv "$RUN_DIR/"
cd "$RUN_DIR" || exit 1

poetry run npx codex exec "Follow the instructions given in 'AGENTS.md'" --dangerously-bypass-approvals-and-sandbox
