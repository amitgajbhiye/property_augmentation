#!/bin/bash --login

#SBATCH --job-name=dbV3Filter

#SBATCH --output=logs/ufet_exp/entropy_ccvocab/out_1stcluster_thresh50_75_ufet_mcrae_cslb_deberta_v3_large_filter_sim_props_thres50_propfix_contrastive.txt
#SBATCH --error=logs/ufet_exp/entropy_ccvocab/err_1stcluster_thresh50_75_ufet_mcrae_cslb_deberta_v3_large_filter_sim_props_thres50_propfix_contrastive.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --mem=18G
#SBATCH --gres=gpu:1

#SBATCH --time 0-04:30:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/6_mcrae_cslb_deberta_v3_large_filter_complementary_clusters_filterthres50_contrastive_propfix.json
python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/6_mcrae_cslb_deberta_v3_large_filter_complementary_clusters_filterthres75_contrastive_propfix.json

echo 'Job Finished!'
