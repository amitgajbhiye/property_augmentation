#!/bin/bash --login

#SBATCH --job-name=McRae25Emb

#SBATCH --output=logs/ufet_exp/entropy_ccvocab/out_for_svm_con_embeddings_mcrae25.txt
#SBATCH --error=logs/ufet_exp/entropy_ccvocab/err_for_svm_con_embeddings_mcrae25.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10G
#SBATCH --gres=gpu:1

#SBATCH -t 0-0:40:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/for_svm_con_embeddings_mcrae25.json
echo 'finished!'