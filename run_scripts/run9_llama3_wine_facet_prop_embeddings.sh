#!/bin/bash --login

#SBATCH --job-name=llama3_facet_prop

#SBATCH --output=logs/embeds_for_commonalities/out_2_llama3_wine_facet_property_bienc_entropy_bert_large_cnetpchatgpt.txt
#SBATCH --error=logs/embeds_for_commonalities/err_2_llama3_wine_facet_property_bienc_entropy_bert_large_cnetpchatgpt.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=12G
#SBATCH -t 0-01:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/commonality_llama/2_llama3_wine_facet_property_bienc_entropy_bert_large_cnetpchatgpt.json

echo 'Job Finished !!!'
