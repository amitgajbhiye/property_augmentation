#!/bin/bash --login

#SBATCH --job-name=ufet_cnetP_vocab_cnetp_bert_large_bienc

#SBATCH --output=logs/ufet_exp/ce_cnetp_vocab_cnetp_bienc/out_get_main_clusters.txt
#SBATCH --error=logs/ufet_exp/ce_cnetp_vocab_cnetp_bienc/err_get_main_clusters.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-4:00:00

conda activate venv


python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_vocab_cnetp_bienc/1_ufettype_embed.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_vocab_cnetp_bienc/2_cnetp_vocab_embeds.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_vocab_cnetp_bienc/3_ufet_type_similar_50cnetp_props.json

python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_vocab_cnetp_bienc/4_mcrae_cslb_deb3l_filter_con_sim_50props_filterthresh50_75_90.json

echo 'Job Finished !!!'
