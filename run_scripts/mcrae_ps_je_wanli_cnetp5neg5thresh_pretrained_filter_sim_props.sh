#!/bin/bash --login

#SBATCH --job-name=propFilter_wanli_cnetp5neg5thresh

#SBATCH --output=logs/data_sampling/out_mcrae_ps_je_wanli_cnetp5neg5thresh_pretrained_filter_sim_props.txt
#SBATCH --error=logs/data_sampling/err_mcrae_ps_je_wanli_cnetp5neg5thresh_pretrained_filter_sim_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:30:00


echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/mcrae_ps_je_wanli_cnetp5neg5thresh_pretrained_filter_sim_props.json


echo 'Job Finished!'
