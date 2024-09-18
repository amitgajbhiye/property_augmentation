#!/bin/bash --login

#SBATCH --job-name=wikidata_bienc_bert_large

#SBATCH --output=logs/wikidata/out_wikidata_bienc_bert_large.txt
#SBATCH --error=logs/wikidata/err_wikidata_bienc_bert_large.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-02:00:00

conda activate venv

python3 for_llm_ranking_get_embeds_and_train_data.py --config_file configs/wikidata/2_bienc_entropy_bert_large_mscggkb.json
python3 for_llm_ranking_get_embeds_and_train_data.py --config_file configs/wikidata/3_bienc_entropy_bert_large_cnetpchatgpt.json

echo 'Job Finished !!!'
