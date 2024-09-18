#!/bin/bash --login

#SBATCH --job-name=ufetConSim50VocabProps

#SBATCH --output=logs/ufet_exp/out_ufet_bert_large_get_con_similar_50cnet_vocab_props.txt
#SBATCH --error=logs/ufet_exp/err_ufet_ufet_bert_large_get_con_similar_50cnet_vocab_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --gres=gpu:1

#SBATCH -p compute
#SBATCH --mem=16G
#SBATCH -t 0-02:00:00


conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_bert_large_get_con_similar_50cnet_props.json

echo 'Job Finished!'
