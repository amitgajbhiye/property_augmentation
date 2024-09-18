#!/bin/bash --login

#SBATCH --job-name=csft_wanli_cnetp5neg50thresh_concept_split

#SBATCH --output=logs/mcrae_finetune/out_cs_wanli_cnetp5neg50thresh_prop_conj_je_mcrae_fintune.txt
#SBATCH --error=logs/mcrae_finetune/err_cs_wanli_cnetp5neg50thresh_prop_conj_je_mcrae_fintune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 0-4:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py  --config_file configs/3_finetune/cs_wanli_cnetp5neg50thresh_prop_conj_je_mcrae_fintune.json

echo 'Job Finished !!!'
