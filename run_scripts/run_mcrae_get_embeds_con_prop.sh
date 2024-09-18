#!/bin/bash --login

#SBATCH --job-name=McRconPropEmbed

#SBATCH --output=logs/embeddings/out_mcrae_get_embeds_concepts_and_properties.txt
#SBATCH --error=logs/embeddings/err_mcrae_get_embeds_concepts_and_properties.txt

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

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_get_embeds_concepts.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_get_embeds_property.json

echo 'Job Finished!'