#!/bin/bash

slurm_script="scripts/run_pairwise_eval.sh"

# datasets=("affairs" "caschools" "crofoot" "hurricane" "reading" "amtl" "mortgage" "soccer" "boxes" "fish" "panda_nuts" "teachingratings")
datasets=("amtl" "crofoot" "hurricane" "mortgage")

for dataset in "${datasets[@]}"; do
    echo "Submitting eval job for dataset: $dataset"
    sbatch $slurm_script $dataset # submit SLURM job using the specified script
done
