#!/bin/bash --login

#SBATCH --job-name=psWanliCnet5Neg5threshpropConjData

#SBATCH --output=logs/data_sampling/out_mcrae_ps_get_predict_prop_sim_props_wanli_5neg50thres.txt
#SBATCH --error=logs/data_sampling/err_mcrae_ps_get_predict_prop_sim_props_wanli_5neg50thres.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

##SBATCH --partition gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH --partition compute
#SBATCH --time 0-2:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/5_get_predict_prop_sim_data/mcrae_ps_get_predict_prop_sim_props_wanli_5neg50thres.json


echo 'Job Finished!'
