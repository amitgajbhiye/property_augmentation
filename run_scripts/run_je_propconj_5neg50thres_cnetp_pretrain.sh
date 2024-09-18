#!/bin/bash --login

#SBATCH --job-name=Ptr5neg50th

#SBATCH --output=logs/pretrain/out_je_propconj_5neg50thres_cnetp_pretrain.txt
#SBATCH --error=logs/pretrain/err_je_propconj_5neg50thres_cnetp_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH --qos="gpu7d"
#SBATCH -t 6-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/1_configs_pretrain/je_propconj_bb_cnetp_5neg.json

echo 'Job Finished!'