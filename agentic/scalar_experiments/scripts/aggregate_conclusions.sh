#!/bin/bash
#SBATCH --job-name=aggregate_conclusions
#SBATCH --output=slurm_output/%x-%j.out
#
#SBATCH --partition=low
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --ntasks=1

# Run from agentic/scalar_experiments/
set -euo pipefail

mkdir -p slurm_output

# Fix broken conclusions first (programmatic + LLM fallback)
bash scripts/fix-conclusions.sh

poetry run python scripts/aggregate_conclusions.py
