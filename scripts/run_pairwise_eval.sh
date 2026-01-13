#!/bin/bash
#SBATCH --job-name=pairwise_eval
#SBATCH -o ./out/pairwise_eval_%j.log  
#SBATCH -e ./out/pairwise_eval_%j.err
#
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#
#SBATCH -p yugroup
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1

mkdir -p out

# cd to project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cd "$SCRIPT_DIR/.." || exit 1
fi

if ! command -v poetry &> /dev/null; then
    echo "Error: poetry not found"
    exit 1
fi

if [ ! -f "pyproject.toml" ]; then
    echo "Error: not in project root"
    exit 1
fi

# load .env from project root (if present) and export its variables
if [ -f ".env" ]; then
    set -o allexport
    # shellcheck source=/dev/null
    source ".env"
    set +o allexport
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

DATASET="mortgage"
NUM_MULTIRUNS=5
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"
ANALYSIS_BASE_DIR="outputs/analysis"

echo "Job ID: ${SLURM_JOB_ID}"

START_TIME=$(date +%s)
ARGS=(
    --dataset "$DATASET"
    --num-multiruns "$NUM_MULTIRUNS"
    --llm-provider "$LLM_PROVIDER"
    --llm-model "$LLM_MODEL"
)
if [ -n "$ANALYSIS_BASE_DIR" ]; then
    ARGS+=(--analysis-base-dir "$ANALYSIS_BASE_DIR")
fi
poetry run python scripts/run_pairwise_eval.py "${ARGS[@]}"
EXIT_CODE=$?
END_TIME=$(date +%s)

DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"

exit $EXIT_CODE

