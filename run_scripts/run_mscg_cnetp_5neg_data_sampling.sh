#!/bin/bash --login

#SBATCH --job-name=5negDataMSCG_CNETP

#SBATCH --output=logs/data_sampling/out_mscg_cnetp_neg_data_sampling_for_conprop_je_enc.txt
#SBATCH --error=logs/data_sampling/err_mscg_cnetp_neg_data_sampling_for_conprop_je_enc.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH -t 0-10:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 src/data_prepr/je_con_prop_negative_data_sampling.py

echo 'Job Finished!'