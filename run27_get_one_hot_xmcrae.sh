#!/bin/bash --login

#SBATCH --job-name=xmcrae1Hot

#SBATCH --output=logs/classification_vocab/xmcrae/out_get_one_hot_enc.txt
#SBATCH --error=logs/classification_vocab/xmcrae/err_get_one_hot_enc.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem

#SBATCH --mem=20G
#SBATCH -t 0-05:00:00

conda activate venv

python3 one_hot_converter.py data/classification_vocab/xmcrae data/classification_vocab/xmcrae/onehot_encodings

echo 'Job Finished !!!'