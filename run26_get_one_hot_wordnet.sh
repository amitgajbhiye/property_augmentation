#!/bin/bash --login

#SBATCH --job-name=WordNet1Hot

#SBATCH --output=logs/classification_vocab/wordnet/out_get_one_hot_enc.txt
#SBATCH --error=logs/classification_vocab/wordnet/err_get_one_hot_enc.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=125G
#SBATCH -t 0-12:00:00

conda activate venv

python3 one_hot_converter.py data/classification_vocab/wordnet data/classification_vocab/wordnet/onehot_encodings

echo 'Job Finished !!!'