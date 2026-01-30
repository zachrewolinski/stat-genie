#!/bin/bash
#
# create toy/run1/, copy in instructions and data, then run codex there.
# outputs (analysis.py, conclusion.txt, etc.) go into toy/run1/.
# this is done so that we don't have to change the code to run the analysis on a single run.

set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOY_DIR="${EXPERIMENTS_DIR}/toy"
RUN_DIR="${TOY_DIR}/run1"

cd "$EXPERIMENTS_DIR"
mkdir -p "$RUN_DIR"
cp "$TOY_DIR"/AGENTS.md "$TOY_DIR"/info.json "$TOY_DIR"/compas.csv "$RUN_DIR/"
cd "$RUN_DIR" || exit 1

poetry run npx codex exec "Follow the instructions given in 'AGENTS.md'" --dangerously-bypass-approvals-and-sandbox
