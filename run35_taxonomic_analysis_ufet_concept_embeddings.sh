#!/bin/bash --login

#SBATCH --job-name=ufetTypeEmbeds

#SBATCH --output=logs/taxonomic_analysis/out_taxonomic_analysis_ufet_embeds.txt
#SBATCH --error=logs/taxonomic_analysis/err_taxonomic_analysis_ufet_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-5:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/taxonomic_analysis/1_ufettype_embed.json

echo 'Job Finished !!!'
