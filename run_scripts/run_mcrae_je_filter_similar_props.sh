#!/bin/bash --login

#SBATCH --job-name=conSimProps

#SBATCH --output=logs/data_sampling/out_mcrae_5_10neg_50_75_80thres_je_filter_con_sim_vocab_props.txt
#SBATCH --error=logs/data_sampling/err_mcrae_5_10neg_50_75_80thres_je_filter_con_sim_vocab_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH -t 0-02:00:00


echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/mcrae_5neg80thres_je_filter_sim_props.json
python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/mcrae_10neg50thres_je_filter_sim_props.json
python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/mcrae_10neg75thres_je_filter_sim_props.json

echo 'Job Finished!'