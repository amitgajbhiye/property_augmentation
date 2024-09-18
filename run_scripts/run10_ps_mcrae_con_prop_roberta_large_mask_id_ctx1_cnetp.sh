#!/bin/bash --login

#SBATCH --job-name=WanliRLPSMcFT

#SBATCH --output=logs/mcrae_finetune/out_ps_mcrae_con_prop_wanli_pretrained_roberta_large_mask_id_ctx1_cnetp.txt
#SBATCH --error=logs/mcrae_finetune/err_ps_mcrae_con_prop_wanli_pretrained_roberta_large_mask_id_ctx1_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 0-07:00:00

conda activate venv

python3 model/lm_con_prop.py --config_file configs/3_finetune/ps_mcrae_con_prop_wanli_pretrained_roberta_large_mask_id_ctx1_cnetp.json

echo 'Job Finished!'