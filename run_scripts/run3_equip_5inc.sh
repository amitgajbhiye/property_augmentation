#!/bin/bash --login

#SBATCH --job-name=equipBienc

#SBATCH --output=logs/commonality_eval_taxo/out_pretrain_bienc_bl_llama3_equipment_taxo_5inc_con_fac_colon_prop.txt
#SBATCH --error=logs/commonality_eval_taxo/err_pretrain_bienc_bl_llama3_equipment_taxo_5inc_con_fac_colon_prop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

#SBATCH --time 2-00:00:00

conda activate llm_prompts
    
python3 bienc_run_model.py --config_file configs/commonality_bienc_eval_taxo/pretrain_bienc_bl_llama3_equipment_taxo_5inc_con_fac_colon_prop.json

echo 'Job Finished!'
