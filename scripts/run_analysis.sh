#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH -o ./out/analysis_%j.log  
#SBATCH -e ./out/analysis_%j.err
#
#SBATCH --mail-user=austin.zane@berkeley.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#
#SBATCH -p yugroup
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH -t 1:00:00

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

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

DATASET="mortgage"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"
NUM_RUNS=1

echo "Job ID: ${SLURM_JOB_ID}"

TOTAL_START=$(date +%s)

echo "Running analysis 1 (noperturb)..."
START_TIME=$(date +%s)
poetry run python scripts/run_analysis.py \
    --dataset "$DATASET" \
    --analysis-num 1 \
    --perturbation-type noperturb \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --num-runs "$NUM_RUNS"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Analysis 1 done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo "Running analysis 2 (anonymize)..."
START_TIME=$(date +%s)
poetry run python scripts/run_analysis.py \
    --dataset "$DATASET" \
    --analysis-num 2 \
    --perturbation-type anonymize \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --num-runs "$NUM_RUNS"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Analysis 2 done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo "Running analysis 3 (shuffle_names)..."
START_TIME=$(date +%s)
poetry run python scripts/run_analysis.py \
    --dataset "$DATASET" \
    --analysis-num 3 \
    --perturbation-type shuffle_names \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --num-runs "$NUM_RUNS"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Analysis 3 done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo "All analyses done in ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"

exit $EXIT_CODE

