#!/bin/bash --login

#SBATCH --job-name=ufetC_ContraExp

#SBATCH --output=logs/ufet_exp/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/out_exp_contra_bert_large_cnetp_vocab.txt
#SBATCH --error=logs/ufet_exp/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/err_exp_contra_bert_large_cnetp_vocab.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-04:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/1_ufettype_embed.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/2_cnetp_vocab_embeds.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/3_ufet_type_similar_50cnetpprops.json
python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/4_mcrae_cslb_deb3l_filter_con_sim_50props_filterthresh50_75.json

python3 get_property_similar_concepts.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/5_get_prop_similar50types_filterthresh50.json
python3 get_property_similar_concepts.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/5_get_prop_similar50types_filterthresh75.json
python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/6_mcrae_cslb_deb3l_filter_prop_sim_50cons_filterthresh50.json
python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/contra_cnetp_vocab_pt_conceptfix_contra_bertlarge_cnetpchatgpt100k_model/6_mcrae_cslb_deb3l_filter_prop_sim_50cons_filterthresh75.json


echo 'Job Finished !!!'