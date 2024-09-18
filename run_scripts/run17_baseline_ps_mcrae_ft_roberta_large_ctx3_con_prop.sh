#!/bin/bash --login

#SBATCH --job-name=rlMcPsBL

#SBATCH --output=logs/mcrae_finetune/out_baseline_ps_mcrae_ft_roberta_large_ctx3_con_prop.txt
#SBATCH --error=logs/mcrae_finetune/err_baseline_ps_mcrae_ft_roberta_large_ctx3_con_prop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=18g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 0-05:00:00


conda activate venv

python3 model/lm_con_prop.py --config_file configs/3_finetune/baseline_ps_mcrae_ft_roberta_large_ctx3_con_prop.json

echo 'Job Finished!'
