#!/bin/bash --login

#SBATCH --job-name=JEnegDataPrep

#SBATCH --output=logs/joint_encoder_pretraining_data_prep/out_je_con_prop_pretraining_neg_data_sampling.txt
#SBATCH --error=logs/joint_encoder_pretraining_data_prep/err_je_con_prop_pretraining_neg_data_sampling.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=5g

#SBATCH -t 0-03:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 src/data_prepr/je_con_prop_negative_data_sampling.py

echo 'finished!'