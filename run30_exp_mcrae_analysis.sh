#!/bin/bash --login

#SBATCH --job-name=mc_analysis

#SBATCH --output=logs/mcrae_analysis_exp/out_mcrae_cnetpchatgpt_main_cluster.txt
#SBATCH --error=logs/mcrae_analysis_exp/err_mcrae_cnetpchatgpt_main_cluster.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-03:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/mcrae_analysis_exp/1_get_con_embed.json
python3 get_embeds_and_train_data.py --config_file configs/mcrae_analysis_exp/2_cnetp_chatgpt100k_vocab_embeds.json
python3 get_embeds_and_train_data.py --config_file configs/mcrae_analysis_exp/3_con_similar_50cnetp_chatgpt100k_props.json

python3 je_filter_similar_props.py --config_file configs/mcrae_analysis_exp/4_mcrae_cslb_deb3l_filter_con_sim_50props_filterthresh50_75_90.json

echo 'Job Finished !!!'
