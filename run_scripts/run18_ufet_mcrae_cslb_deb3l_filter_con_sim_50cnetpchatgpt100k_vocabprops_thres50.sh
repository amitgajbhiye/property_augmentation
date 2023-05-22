#!/bin/bash --login

#SBATCH --job-name=dbV3Filter

#SBATCH --output=logs/ufet_exp/entropy_ccvocab/out_4_mcrae_cslb_deb3l_filter_con_sim_50cnetpchatgpt100k_vocabprops_thres50.txt
#SBATCH --error=logs/ufet_exp/entropy_ccvocab/err_4_mcrae_cslb_deb3l_filter_con_sim_50cnetpchatgpt100k_vocabprops_thres50.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

#SBATCH --time 0-02:30:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/4_mcrae_cslb_deb3l_filter_con_sim_50cnetpchatgpt100k_vocabprops_thres50.json

echo 'Job Finished!'
