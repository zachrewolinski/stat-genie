#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=zachrewolinski@berkeley.edu

slurm_script="scripts/run_pairwise_eval.sh"

# datasets=("mortgage" "panda_nuts" "reading" "soccer" "teachingratings")
# datasets=("caschools" "crofoot" "hurricane" "amtl" "mortgage" "soccer" "boxes" "panda_nuts" "teachingratings")
datasets=("amtl")

for dataset in "${datasets[@]}"; do
    echo "Submitting eval job for dataset: $dataset"
    sbatch --wait $slurm_script $dataset # submit SLURM job using the specified script
done
