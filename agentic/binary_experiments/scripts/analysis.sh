#!/bin/bash
# Run from agentic/experiments/ (job cwd must be experiments/).

# the first three inputs are dataset name, perturbation type, and run number.
dataset_name="$1"
perturbation_type="$2"
run_number="$3"

# the fourth input is whether to use claude code or codex
agent_name="$4"

# assert that the agent name is in ["claude_code", "codex"]
if [ "$agent_name" != "claude_code" ] && [ "$agent_name" != "codex" ]; then
    echo "Error: agent name must be either 'claude_code' or 'codex'"
    exit 1
fi

# call make-subdir.py with the given dataset name and perturbation type
poetry run python scripts/make-subdir.py \
    --dataset "$dataset_name" \
    --perturbation-type "$perturbation_type" \
    --run_number "$run_number"

# change directory to the newly created subdirectory
cd "outputs/$dataset_name/$perturbation_type/run$run_number" || exit 1

# if the agent name is claude code, do the following
if [ "$agent_name" == "claude_code" ]; then
    # disable use of cache for claude code
    export DISABLE_PROMPT_CACHING=1

    # run claude code to generate an answer to the research question
    poetry run claude -p "Follow the instructions given in 'instructions.txt'" --allowedTools "Read,Edit,Bash" --dangerously-skip-permissions
fi

# if the agent name is codex, do the following
if [ "$agent_name" == "codex" ]; then
    # run codex to generate an answer to the research question
    poetry run npx codex exec "Follow the instructions given in 'instructions.txt'" --sandbox workspace-write
fi

