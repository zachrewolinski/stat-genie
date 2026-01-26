#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=zachrewolinski@berkeley.edu

slurm_script="scripts/run_analysis.sh"

datasets=("caschools" "crofoot" "hurricane" "reading" "amtl" "mortgage" "soccer" "boxes" "panda_nuts" "teachingratings")
# datasets=("boxes" "fish" "panda_nuts" "teachingratings" "caschools" "reading")
# datasets=("hurricane" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")
# datasets=("affairs")

for dataset in "${datasets[@]}"; do
    echo "Submitting analysis job for dataset: $dataset"
    sbatch --wait $slurm_script $dataset # submit SLURM job using the specified script
done
