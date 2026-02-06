#!/bin/bash

set -euo pipefail

# change directory to the newly created subdirectory
cd "toy/run1" || exit 1

echo "working directory:"
pwd

# run codex to generate an answer to the research question
poetry run npx codex exec "Follow the instructions given in 'AGENTS.md'" --sandbox workspace-write