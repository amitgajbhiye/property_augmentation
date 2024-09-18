#!/bin/bash --login

#SBATCH --job-name=5nCnetpMCcsFt_con_prop

#SBATCH --output=logs/mcrae_finetune/out_cs_mcrae_5negcnetp_con_prop_pretrained_finetune.txt
#SBATCH --error=logs/mcrae_finetune/err_cs_mcrae_5negcnetp_con_prop_pretrained_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 0-4:30:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_con_prop.py --config_file configs/3_finetune/cs_mcrae_5negcnetp_con_prop_pretrained_finetune.json

echo 'Job Finished !!!'
