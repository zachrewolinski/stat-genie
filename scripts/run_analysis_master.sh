#!/bin/bash

slurm_script="scripts/run_analysis.sh"

# datasets=("affairs" "caschools" "crofoot" "hurricane" "reading" "amtl" "mortgage" "soccer" "boxes" "fish" "panda_nuts" "teachingratings")
# datasets=("boxes" "fish" "panda_nuts" "teachingratings" "caschools" "reading")
datasets=("panda_nuts")

for dataset in "${datasets[@]}"; do
    echo "Submitting analysis job for dataset: $dataset"
    sbatch $slurm_script $dataset # submit SLURM job using the specified script
done
