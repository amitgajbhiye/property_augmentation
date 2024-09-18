#!/bin/bash --login

#SBATCH --job-name=psFtMCcnet

#SBATCH --output=logs/mcrae_finetune/out_ps_ft_mcrae_ps_cnetp_bienc.txt
#SBATCH --error=logs/mcrae_finetune/err_ps_ft_mcrae_ps_cnetp_bienc.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

##SBATCH --qos=gpu7d
#SBATCH --time 0-10:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/ps_ft_mcrae_ps_cnetp_bienc.json

echo 'Job Finished!'