#!/bin/bash --login

#SBATCH --job-name=bbPretrainCnetP

#SBATCH --output=logs/pretrain/out_berta_base_je_pretrain_mask_id_ctx1_con_prop_cnetp.txt
#SBATCH --error=logs/pretrain/err_berta_base_je_pretrain_mask_id_ctx1_con_prop_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=9g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 2-00:00:00


conda activate venv

python3 model/lm_con_prop.py --config_file configs/1_configs_pretrain/berta_base_je_pretrain_mask_id_ctx1_con_prop_cnetp.json

echo 'Job Finished!'