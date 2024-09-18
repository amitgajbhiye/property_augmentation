#!/bin/bash --login

#SBATCH --job-name=NegDSamp

#SBATCH --output=logs/data_sampling/out_je_conprop_5_10_20_neg_new_train_valid_data_sampling.txt
#SBATCH --error=logs/data_sampling/err_je_conprop_5_10_20_neg_new_train_valid_data_sampling.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --mem=6gs
##SBATCH --gres=gpu:1

#SBATCH -p highmem
#SBATCH -t 0-13:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 src/data_prepr/je_con_prop_negative_data_sampling.py

echo 'Job Finished !!!'