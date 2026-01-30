#!/bin/bash

# change directory to the newly created subdirectory
cd "toy" || exit 1

# run codex to generate an answer to the research question
poetry run npx codex exec "Follow the instructions given in 'AGENTS.md'" --dangerously-bypass-approvals-and-sandbox
