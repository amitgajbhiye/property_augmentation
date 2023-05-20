#!/bin/bash --login

#SBATCH --job-name=ufetPropSimilar50Cons

#SBATCH --output=logs/ufet_exp/entropy_ccvocab/out_5_complementary_clusters_debv3l_filterthresh50_75_ufet_contrastive_propfix_bert_large_get_property_similar_50concepts.txt
#SBATCH --error=logs/ufet_exp/entropy_ccvocab/err_5_complementary_clusters_debv3l_filterthresh50_75_ufet_contrastive_propfix_bert_large_get_property_similar_50concepts.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-4:00:00


conda activate venv

python3 get_property_similar_concepts.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/5_complementary_clusters_debv3l_filterthresh50_ufet_contrastive_propfix_bert_large_get_property_similar_50concepts.json
python3 get_property_similar_concepts.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/5_complementary_clusters_debv3l_filterthresh75_ufet_contrastive_propfix_bert_large_get_property_similar_50concepts.json

echo 'Job Finished !!!'
