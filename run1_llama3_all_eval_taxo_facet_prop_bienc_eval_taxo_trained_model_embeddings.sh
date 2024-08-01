#!/bin/bash --login

#SBATCH --job-name=taxoBiencsEmbed

#SBATCH --output=logs/embeds_for_commonalities/out_run1_llama3_science_and_food_eval_taxo_facet_prop_bienc_eval_taxo_trained_model_embeddings.txt
#SBATCH --error=logs/embeds_for_commonalities/err_run1_llama3_science_and_food_eval_taxo_facet_prop_bienc_eval_taxo_trained_model_embeddings.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=12G
#SBATCH -t 0-04:00:00

conda activate venv

# python3 get_embeds_and_train_data.py --config_file configs/commonality_bienc_eval_taxo/get_embed_llama3_commonsense_facet_property_bienc_commonsense_taxo_trained.json
# python3 get_embeds_and_train_data.py --config_file configs/commonality_bienc_eval_taxo/get_embed_llama3_environment_facet_property_bienc_environment_taxo_trained.json
# python3 get_embeds_and_train_data.py --config_file configs/commonality_bienc_eval_taxo/get_embed_llama3_equipment_facet_property_bienc_equipment_taxo_trained.json

python3 get_embeds_and_train_data.py --config_file configs/commonality_bienc_eval_taxo/get_embed_llama3_science_facet_property_bienc_science_taxo_trained.json
python3 get_embeds_and_train_data.py --config_file configs/commonality_bienc_eval_taxo/get_embed_llama3_food_facet_property_bienc_food_taxo_trained.json

echo 'Job Finished !!!'
