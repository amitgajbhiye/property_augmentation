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

python3 je_filter_similar_props.py --config_file configs/mcrae_analysis_exp/prop_similar_analysis/4_mcrae_cslb_deb3l_filter_con_sim_50props_filterthresh50_75_90.json

echo 'Job Finished !!!'
