#!/bin/bash --login

#SBATCH --job-name=conOnlyPropConjData

#SBATCH --output=logs/data_sampling/out_get_pretrain_cnetp_con_only_sim_prop_conj_data.txt
#SBATCH --error=logs/data_sampling/err_get_pretrain_cnetp_con_only_sim_prop_conj_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

##SBATCH --partition gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH --partition compute
#SBATCH --time 0-10:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/6_get_pretrain_cnetp_con_only_sim_prop_conj_data.json

echo 'Job Finished!'