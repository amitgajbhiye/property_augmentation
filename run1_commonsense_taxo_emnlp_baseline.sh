#!/bin/bash --login

#SBATCH --job-name=comm_emnlp_baseline

#SBATCH --output=logs/commonality_emnlp_baseline_exp/out_commonsense_taxo_emnlp_baseline.txt
#SBATCH --error=logs/commonality_emnlp_baseline_exp/err_commonsense_taxo_emnlp_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH --mem=32G
#SBATCH -t 0-03:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/commonality_evaltaxo_emnlp_baseline_exp/commonsense/1_get_concept_embeds.json
python3 get_embeds_and_train_data.py --config_file configs/commonality_evaltaxo_emnlp_baseline_exp/commonsense/2_get_cnet_chatgpt_vocab_embeds.json
python3 get_embeds_and_train_data.py --config_file configs/commonality_evaltaxo_emnlp_baseline_exp/commonsense/3_concept_similar_50cnetpprops.json
python3 je_filter_similar_props.py --config_file configs/commonality_evaltaxo_emnlp_baseline_exp/commonsense/4_mcrae_cslb_deb3l_filter_con_sim_50props_filterthresh50.json

echo 'Job Finished !!!'