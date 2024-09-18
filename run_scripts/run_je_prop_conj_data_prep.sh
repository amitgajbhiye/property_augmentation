#!/bin/bash --login

#SBATCH --job-name=JEdataPrep

#SBATCH --output=logs/get_con_pro_embeddings/out_6_je_prop_conj_data_prep.txt
#SBATCH --error=logs/get_con_pro_embeddings/err_6_je_prop_conj_data_prep.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p htc
#SBATCH --mem=5g

#SBATCH -t 0-03:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 src/data_prepr/je_prop_conjuction_pretraining_data_prepration.py

echo 'finished!'