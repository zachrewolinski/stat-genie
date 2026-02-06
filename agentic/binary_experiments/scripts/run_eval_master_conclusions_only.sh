#!/bin/bash

slurm_script="scripts/run_pairwise_eval_conclusions_only.sh"

datasets=("amtl" "boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")
# datasets=("affairs" "amtl" "boxes" "caschools" "crofoot" "hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")

for dataset in "${datasets[@]}"; do
    echo "Submitting conclusions-only eval job for dataset: $dataset"
    sbatch --wait $slurm_script $dataset
done
