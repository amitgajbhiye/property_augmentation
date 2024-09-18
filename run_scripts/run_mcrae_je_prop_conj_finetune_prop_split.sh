#!/bin/bash --login

#SBATCH --job-name=bs8FtPSMc

#SBATCH --output=logs/mcrae_finetune/out_mcrae_ft_ps_5neg50thres_prop_conj_je.txt
#SBATCH --error=logs/mcrae_finetune/err_mcrae_ft_ps_5neg50thres_prop_conj_je.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 0-3:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/3_finetune/ps_5neg50thres_prop_conj_je_mcrae_fintune.json


echo 'Job Finished !!!'