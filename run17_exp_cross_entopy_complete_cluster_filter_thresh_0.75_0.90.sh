#!/bin/bash --login

#SBATCH --job-name=ufet_CE_complete_cluster

#SBATCH --output=logs/ufet_exp/ce_cnetp_chatgpt_ce_model_complete_clusters/out_get_clusters.txt
#SBATCH --error=logs/ufet_exp/ce_cnetp_chatgpt_ce_model_complete_clusters/err_get_clusters.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-12:00:00

conda activate venv


python3 get_property_similar_concepts.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_ce_model_complete_clusters/5_get_prop_similar50types_filterthresh75.json
python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_ce_model_complete_clusters/6_mcrae_cslb_deb3l_filter_prop_sim_50cons_filterthresh75.json


python3 get_property_similar_concepts.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_ce_model_complete_clusters/5_get_prop_similar50types_filterthresh90.json
python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_ce_model_complete_clusters/6_mcrae_cslb_deb3l_filter_prop_sim_50cons_filterthresh90.json


echo 'Job Finished !!!'
