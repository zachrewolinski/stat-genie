#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH -o ./out/analysis_%j.log  
#SBATCH -e ./out/analysis_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1

# Mail user is provided at submission time so different users can set their
# OWN email in a .env file and use the provided `scripts/submit_run_analysis.sh`
# wrapper which passes `--mail-user` to `sbatch`.


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

# accept dataset as first positional arg (sbatch run_analysis.sh <dataset>)
# or from exported DATASET env var; default to "caschools" if neither provided.
if [ -n "$1" ]; then
    DATASET="$1"
elif [ -z "$DATASET" ]; then
    DATASET="caschools"
fi
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"
NUM_RUNS=5

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

echo "Running analysis 4 (add_features)..."
START_TIME=$(date +%s)
poetry run python scripts/run_analysis.py \
    --dataset "$DATASET" \
    --analysis-num 4 \
    --perturbation-type add_features \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --num-runs "$NUM_RUNS"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Analysis 4 done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo "Running analysis 5 (replace_with_rvs)..."
START_TIME=$(date +%s)
poetry run python scripts/run_analysis.py \
    --dataset "$DATASET" \
    --analysis-num 5 \
    --perturbation-type replace_with_rvs \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --num-runs "$NUM_RUNS"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Analysis 5 done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo "Running analysis 6 (positive_leading_statement)..."
START_TIME=$(date +%s)
poetry run python scripts/run_analysis.py \
    --dataset "$DATASET" \
    --analysis-num 6 \
    --perturbation-type positive_leading_statement \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --num-runs "$NUM_RUNS"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Analysis 6 done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo "Running analysis 7 (negative_leading_statement)..."
START_TIME=$(date +%s)
poetry run python scripts/run_analysis.py \
    --dataset "$DATASET" \
    --analysis-num 7 \
    --perturbation-type negative_leading_statement \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --num-runs "$NUM_RUNS"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Analysis 7 done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"
if [ $EXIT_CODE -ne 0 ]; then
    exit $EXIT_CODE
fi

echo "Running analysis 8 (replace_and_positive_statement)..."
START_TIME=$(date +%s)
poetry run python scripts/run_analysis.py \
    --dataset "$DATASET" \
    --analysis-num 8 \
    --perturbation-type replace_and_positive_statement \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --num-runs "$NUM_RUNS"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Analysis 8 done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"
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

