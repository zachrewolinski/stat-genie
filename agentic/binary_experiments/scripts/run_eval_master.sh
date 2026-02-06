#!/bin/bash

slurm_script="scripts/run_pairwise_eval.sh"

datasets=("affairs" "amtl" "boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")

for dataset in "${datasets[@]}"; do
    echo "Submitting eval job for dataset: $dataset"
    sbatch --wait $slurm_script $dataset # submit SLURM job using the specified script
done
