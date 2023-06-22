#!/bin/bash --login

#SBATCH --job-name=taxoAna

#SBATCH --output=logs/taxonomic_analysis/out_taxonomic_analysis_ufet_con_similar_cons.txt
#SBATCH --error=logs/taxonomic_analysis/err_taxonomic_analysis_ufet_con_similar_cons.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem

#SBATCH --mem=32G
#SBATCH -t 0-3:00:00

conda activate venv

python3 taxonomic_analysis.py

echo 'Job Finished !!!'
