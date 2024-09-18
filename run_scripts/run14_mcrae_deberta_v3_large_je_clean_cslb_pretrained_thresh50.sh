#!/bin/bash --login

#SBATCH --job-name=filterCon50Sim 

#SBATCH --output=logs/mcrae_clean_vocab_data/out_mcrae_deberta_v3_large_je_clean_cslb_pretrained_thresh50.txt
#SBATCH --error=logs/mcrae_clean_vocab_data/err_mcrae_deberta_v3_large_je_clean_cslb_pretrained_thresh50.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1
##SBATCH --exclusive

#SBATCH --mem=32g
#SBATCH -t 0-02:00:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_data_clean_vocab/mcrae_deberta_v3_large_je_clean_cslb_pretrained_thresh50.json

echo 'Job Finished!'