#!/bin/bash --login

#SBATCH --job-name=mc_analysis

#SBATCH --output=logs/mcrae_analysis_exp/out_run34_exp_mcrae_analysis_deb_cslb.txt
#SBATCH --error=logs/mcrae_analysis_exp/err_run34_exp_mcrae_analysis_deb_cslb.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-03:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/mcrae_analysis_exp/con_similar_analysis_deb_cslb/1_get_con_embed.json
python3 get_embeds_and_train_data.py --config_file configs/mcrae_analysis_exp/con_similar_analysis_deb_cslb/2_cnetp_chatgpt100k_vocab_embeds.json
python3 get_embeds_and_train_data.py --config_file configs/mcrae_analysis_exp/con_similar_analysis_deb_cslb/3_con_similar_50cnetp_chatgpt100k_props.json

python3 je_filter_similar_props.py --config_file configs/mcrae_analysis_exp/con_similar_analysis_deb_cslb/4_cslb_deb3l_filter_con_sim_50props_filterthresh50_75_90.json

echo 'Job Finished !!!'
