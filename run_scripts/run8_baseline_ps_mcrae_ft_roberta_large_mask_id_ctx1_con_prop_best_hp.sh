#!/bin/bash --login

#SBATCH --job-name=rlMcPsBL

#SBATCH --output=logs/mcrae_finetune/out_baseline_ps_mcrae_ft_roberta_large_mask_id_ctx1_con_prop_best_hp.txt
#SBATCH --error=logs/mcrae_finetune/err_baseline_ps_mcrae_ft_roberta_large_mask_id_ctx1_con_prop_best_hp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 0-05:00:00


conda activate venv

python3 model/lm_con_prop.py --config_file configs/3_finetune/baseline_ps_mcrae_ft_roberta_large_mask_id_ctx1_con_prop_best_hp.json

echo 'Job Finished!'
