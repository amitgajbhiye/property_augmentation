#!/bin/bash --login

#SBATCH --job-name=CTX3mcFT

#SBATCH --output=logs/mcrae_finetune/out_ps_mcrae_con_prop_roberta_base_ctx3_cnetp.txt
#SBATCH --error=logs/mcrae_finetune/err_ps_mcrae_con_prop_roberta_base_ctx3_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=9g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 0-06:00:00


conda activate venv

python3 model/lm_con_prop.py --config_file configs/3_finetune/ps_mcrae_con_prop_roberta_base_ctx3_cnetp.json

echo 'Job Finished!'