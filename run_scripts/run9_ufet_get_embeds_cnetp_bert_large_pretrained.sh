#!/bin/bash --login

#SBATCH --job-name=ufetEmedBL

#SBATCH --output=logs/ufet_exp/out_ufet_get_embeds_types_and_cnet_prop_vocab.txt
#SBATCH --error=logs/ufet_exp/err_ufet_get_embeds_types_and_cnet_prop_vocab.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --gres=gpu:1

#SBATCH -p htc
#SBATCH --mem=16G
#SBATCH -t 0-02:00:00


conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_get_embed_bert_large_type_concepts_cnetp_pretrained.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_get_embed_bert_large_cnet_prop_vocab_cnetp_pretrained.json

echo 'Job Finished!'
