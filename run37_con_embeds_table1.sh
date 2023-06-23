#!/bin/bash --login

#SBATCH --job-name=table1ConEmbeds

#SBATCH --output=logs/taxonomic_analysis/out_run37_con_embeds_table1.txt
#SBATCH --error=logs/taxonomic_analysis/err_run37_con_embeds_table1.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-01:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/taxonomic_analysis/1_cons_for_table1.json

echo 'Job Finished !!!'
