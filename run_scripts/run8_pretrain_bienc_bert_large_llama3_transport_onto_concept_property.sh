#!/bin/bash --login

#SBATCH --job-name=cp

#SBATCH --output=logs/llama3_bienc/out_pretrain_bienc_bert_large_llama3_transport_onto_concept_property.txt
#SBATCH --error=logs/llama3_bienc/err_pretrain_bienc_bert_large_llama3_transport_onto_concept_property.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

#SBATCH --time 2-00:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/commonality_llama/pretrain_bienc_bert_large_llama3_transport_onto_concept_property.json

echo 'Job Finished!'
