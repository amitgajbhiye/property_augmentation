#!/bin/bash --login

#SBATCH --job-name=sim_vocab

#SBATCH --output=logs/embeddings/out_concept_embeddings_word_similarity_benchmarks.txt
#SBATCH --error=logs/embeddings/err_concept_embeddings_word_similarity_benchmarks.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:' hostname

conda activate venv


python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/word_similarity_get_concepts_embeddings.json

echo 'Job finished!'