#!/bin/bash
#SBATCH --job-name=pairwise_eval
#SBATCH -o ./out/pairwise_eval_%j.log  
#SBATCH -e ./out/pairwise_eval_%j.err
#
#SBATCH --mail-user=austin.zane@berkeley.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#
#SBATCH -p yugroup
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH -t 0:30:00

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

echo "Running pairwise eval (mortgage, 1 multirun, openai/gpt-5-mini)"
echo "Job ID: ${SLURM_JOB_ID}"

START_TIME=$(date +%s)
poetry run python scripts/run_pairwise_eval.py
EXIT_CODE=$?
END_TIME=$(date +%s)

DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Done in ${HOURS}h ${MINUTES}m ${SECONDS}s (exit: $EXIT_CODE)"

exit $EXIT_CODE

