#!/bin/bash
#SBATCH --job-name=calibration_sim
#SBATCH --output=slurm_output/%x-%j.out
#
#SBATCH --partition=low
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1

set -euo pipefail

mkdir -p slurm_output

for agg in mean median; do
    poetry run python scripts/calibration_sim.py \
        --R 1000 --B 2000 --seed 42 --agg "$agg" --n-jobs -1
done
