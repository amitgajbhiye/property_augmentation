#!/bin/bash --login

#SBATCH --job-name=McPropSimilarAnalysis

#SBATCH --output=logs/mcrae_analysis_exp/con_similar_analysis/out_overlap_count_soretd_deberta_filter_mcrae_analysis.txt
#SBATCH --error=logs/mcrae_analysis_exp/con_similar_analysis/err_overlap_count_sorted_deberta_filter_mcrae_analysis.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem

#SBATCH --mem=32G
#SBATCH -t 0-2:00:00

conda activate venv

python3 mcrae_analysis.py

echo 'Job Finished !!!'
