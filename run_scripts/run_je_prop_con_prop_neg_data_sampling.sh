#!/bin/bash --login

#SBATCH --job-name=JEpcNegSamp

#SBATCH --output=logs/joint_encoder_pretraining_data_prep/out_7_je_prop_con_prop_neg_data_sampling.txt
#SBATCH --error=logs/joint_encoder_pretraining_data_prep/err_7_je_con_prop_pretraining_neg_data_sampling.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=6gs
#SBATCH --gres=gpu:1

#SBATCH -t 0-02:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 src/data_prepr/je_prop_conjuction_pretrain_neg_data_sampling.py

echo 'finished!'