#!/bin/bash --login

#SBATCH --job-name=propVocabEmb

#SBATCH --output=logs/get_embeddings/out_get_embeds_property_vocab_cnetp.txt
#SBATCH --error=logs/get_embeddings/err_get_embeds_property_vocab_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH -p compute
#SBATCH -t 0-02:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/2_get_embeds_property_vocab_cnetp.json


echo 'Job Finished!'