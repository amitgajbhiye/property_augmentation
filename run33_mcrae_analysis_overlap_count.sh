#!/bin/bash --login

#SBATCH --job-name=McPropSimilarAnalysis

#SBATCH --output=logs/mcrae_analysis_exp/con_similar_analysis/out_deberta_filter_mcrae_analysis_overlap_count.txt
#SBATCH --error=logs/mcrae_analysis_exp/con_similar_analysis/err_deberta_filter_mcrae_analysis_overlap_count.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem

#SBATCH --mem=32G
#SBATCH -t 0-20:00:00

conda activate venv

python3 mcrae_analysis.py

echo 'Job Finished !!!'
