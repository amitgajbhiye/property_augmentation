#!/bin/bash --login

#SBATCH --job-name=PosNeg

#SBATCH --output=logs/joint_encoder_pretraining_data_prep/out_joint_enc_rand_and_similar_pos_neg_data_prep.txt
#SBATCH --error=logs/joint_encoder_pretraining_data_prep/err_joint_enc_rand_and_similar_pos_neg_data_prep.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=10g

#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 src/data_prepr/joint_encoder_pretraining_negative_data_sampling.py

echo 'finished!'