#!/bin/bash --login

#SBATCH --job-name=580csMcFT

#SBATCH --output=logs/mcrae_finetune/out_mcrae_concept_split_5neg50thres_je_prop_conj_finetune.txt
#SBATCH --error=logs/mcrae_finetune/err_mcrae_concept_split_5neg50thres_je_prop_conj_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 0-4:30:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/3_finetune/cs_5neg50thres_prop_conj_je_mcrae_fintune.json

echo 'Job Finished !!!'