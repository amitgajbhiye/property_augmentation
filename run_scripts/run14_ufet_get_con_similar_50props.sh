#!/bin/bash --login

#SBATCH --job-name=ufetTypeSimilarPropVocab

#SBATCH --output=logs/ufet_exp/out_ufet_get_type_similar_50centprops.txt
#SBATCH --error=logs/ufet_exp/err_ufet_get_type_similar_50centprops.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --gres=gpu:1

#SBATCH -p htc
#SBATCH --mem=8G
#SBATCH -t 0-03:00:00

echo $SLURM_JOB_ID

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_get_con_similar_50props.json

echo 'Job Finished!'