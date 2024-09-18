#!/bin/bash --login

#SBATCH --job-name=Mask3simPP

#SBATCH --output=logs/pretrain/out_je_pretrain_mask_id_propconj_mscg_cnetp_prdict_prop_2sim.txt
#SBATCH --error=logs/pretrain/err_je_pretrain_mask_id_propconj_mscg_cnetp_prdict_prop_2sim.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute
#SBATCH --mem=10g

##SBATCH --gres=gpu:1
#SBATCH --time 0-01:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/1_configs_pretrain/je_pretrain_mask_id_propconj_mscg_cnetp_prdict_prop_2sim.json

echo 'Job Finished!'
