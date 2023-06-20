#!/bin/bash --login

#SBATCH --job-name=McPropSimilarAnalysis

#SBATCH --output=logs/mcrae_analysis_exp/prop_similar_analysis/out_get_clusters.txt
#SBATCH --error=logs/mcrae_analysis_exp/prop_similar_analysis/err_get_clusters.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-12:00:00

conda activate venv


python3 get_embeds_and_train_data.py --config_file configs/mcrae_analysis_exp/prop_similar_analysis/1_get_mcrae_con_embed.json
python3 get_embeds_and_train_data.py --config_file configs/mcrae_analysis_exp/prop_similar_analysis/2_mcrae_prop_vocab_embeds.json
python3 get_property_similar_concepts.py --config_file configs/mcrae_analysis_exp/prop_similar_analysis/3_get_prop_similar50cons.json

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/cnetp_chatgpt100k_vocab_chatgpt100k_bienc/3_ufet_type_similar_50cnetp_chatgpt100k_props.json

echo 'Job Finished !!!'
