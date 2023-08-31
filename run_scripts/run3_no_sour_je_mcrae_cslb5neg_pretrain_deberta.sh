#!/bin/bash --login

#SBATCH --job-name=NoSourJEdeb_Mcrae_CSLB5neg_pretrain

#SBATCH --output=logs/je_pretrain/out_je_mcrae_cslb5neg_no_sour_pretrain_deberta_v3_large_mask_id_ctx5_con_prop.txt
#SBATCH --error=logs/je_pretrain/err_je_mcrae_cslb5neg_no_sour_pretrain_deberta_v3_large_mask_id_ctx5_con_prop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH --mem=40g
#SBATCH --time 2-00:00:00

conda activate venv

python3 model/lm_con_prop.py --config_file configs/1_configs_pretrain/je_mcrae_cslb5neg_no_sour_pretrain_deberta_v3_large_mask_id_ctx5_con_prop.json

echo 'Job Finished !!!'
