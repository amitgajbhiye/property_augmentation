#!/bin/bash --login

#SBATCH --job-name=dblWanli5PT

#SBATCH --output=logs/pretrain/out_deberta_v3_large_wanli_pretrain_mask_id_ctx5_premise_hypothesis.txt
#SBATCH --error=logs/pretrain/err_deberta_v3_large_wanli_pretrain_mask_id_ctx5_premise_hypothesis.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH -t 02-00:00:00

conda activate venv

python3 model/lm_con_prop.py --config_file configs/1_configs_pretrain/deberta_v3_large_wanli_pretrain_mask_id_ctx5_premise_hypothesis.json

echo 'Job Finished!'