#!/bin/bash --login

#SBATCH --job-name=ufetTypeConEmbeds

#SBATCH --output=logs/ufet_exp/out_ufet_get_type_concepts_embeds_bb_cnetp_pretrained.txt
#SBATCH --error=logs/ufet_exp/err_ufet_get_type_concepts_embeds_bb_cnetp_pretrained.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --gres=gpu:1

#SBATCH -p htc
#SBATCH --mem=8G
#SBATCH -t 0-02:00:00

echo $SLURM_JOB_ID

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_get_type_concepts_embeds_bb_cnetp_pretrained.json

echo 'Job Finished!'