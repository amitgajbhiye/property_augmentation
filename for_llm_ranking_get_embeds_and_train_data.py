import os
import argparse
import gc


import logging
import os
import pickle
import pandas as pd
import numpy as np

import torch

import nltk
from nltk.stem import WordNetLemmatizer
from utils.functions import (
    set_seed,
    create_model,
    read_config,
    to_cpu,
    mcrae_dataset_and_dataloader,
)
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)

cuda_available = torch.cuda.is_available()

device = torch.device("cuda") if cuda_available else torch.device("cpu")

nltk.data.path.append("/scratch/c.scmag3/nltk_data")


def preprocess_get_embedding_data(config):
    inference_params = config.get("inference_params")
    input_data_type = inference_params["input_data_type"]

    log.info(f"Input Data Type : {input_data_type}")

    if input_data_type == "concept":
        data_df = pd.read_csv(
            inference_params["concept_file"],
            sep="\t",
            header=None,
            keep_default_na=False,
        )

    elif input_data_type == "property":
        data_df = pd.read_csv(
            inference_params["property_file"],
            sep="\t",
            header=None,
            keep_default_na=False,
        )

    elif input_data_type == "concept_and_property":
        data_df = pd.read_csv(
            inference_params["concept_property_file"],
            sep="\t",
            header=None,
            keep_default_na=False,
        )

    num_columns = len(data_df.columns)
    log.info(f"Number of columns in input file : {num_columns}")

    input_data_type = inference_params["input_data_type"]

    if input_data_type == "concept" and num_columns == 1:
        log.info(f"Generating Embeddings for Concepts")
        log.info(f"Number of records : {data_df.shape[0]}")

        data_df.rename(columns={0: "concept"}, inplace=True)

        unique_concepts = data_df["concept"].unique()
        data_df = pd.DataFrame(unique_concepts, columns=["concept"])

        data_df["property"] = "dummy_property"
        data_df["label"] = int(0)

    elif input_data_type == "property" and num_columns == 1:
        log.info("Generating Embeddings for Properties")
        log.info(f"Number of records : {data_df.shape[0]}")

        data_df.rename(columns={0: "property"}, inplace=True)

        unique_properties = data_df["property"].unique()
        data_df = pd.DataFrame(unique_properties, columns=["property"])

        data_df["concept"] = "dummy_concept"
        data_df["label"] = int(0)

    elif input_data_type == "concept_and_property" and num_columns in (2, 3):
        log.info("Generating Embeddings for Concepts and Properties")
        log.info(f"Number of records : {data_df.shape[0]}")

        if num_columns == 2:
            data_df["label"] = int(0)

        data_df.rename(columns={0: "concept", 1: "property", 2: "label"}, inplace=True)

    else:
        raise Exception(
            f"Please Enter a Valid Input data type from: 'concept', 'property' or conncept_and_property. \
            Current 'input_data_type' is: {input_data_type}"
        )

    data_df = data_df[["concept", "property", "label"]]

    log.info(f"Final Data Df")
    log.info(data_df.head(n=20))

    return data_df


def generate_embeddings(config):
    inference_params = config.get("inference_params")

    input_data_type = inference_params["input_data_type"]
    model_params = config.get("model_params")
    dataset_params = config.get("dataset_params")

    model = create_model(model_params)

    best_model_path = inference_params["pretrained_model_path"]

    if cuda_available:
        model.load_state_dict(torch.load(best_model_path))
    else:
        model.load_state_dict(
            torch.load(best_model_path, map_location=torch.device("cpu"))
        )

    model.eval()
    model.to(device)

    log.info(f"The model is loaded from :{best_model_path}")
    log.info(f"The model is loaded on : {device}")

    data_df = preprocess_get_embedding_data(config=config)

    dataset, dataloader = mcrae_dataset_and_dataloader(
        dataset_params, dataset_type="test", data_df=data_df
    )

    con_embedding, prop_embedding = {}, {}
    logits_list = []

    for step, batch in enumerate(dataloader):
        concepts_batch, property_batch = dataset.add_context(batch)

        ids_dict = dataset.tokenize(concepts_batch, property_batch)

        if dataset.hf_tokenizer_name in ("roberta-base", "roberta-large"):
            (
                concept_inp_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
            ) = [val.to(device) for _, val in ids_dict.items()]

            concept_token_type_id = None
            property_token_type_id = None

        else:
            (
                concept_inp_id,
                concept_attention_mask,
                concept_token_type_id,
                property_input_id,
                property_attention_mask,
                property_token_type_id,
            ) = [val.to(device) for _, val in ids_dict.items()]

        with torch.no_grad():
            concept_embedding, property_embedding, logits = model(
                concept_input_id=concept_inp_id,
                concept_attention_mask=concept_attention_mask,
                concept_token_type_id=concept_token_type_id,
                property_input_id=property_input_id,
                property_attention_mask=property_attention_mask,
                property_token_type_id=property_token_type_id,
            )

            # print("shape concept_pair_embedding: ", concept_pair_embedding.shape)
            # print("shape relation_embedding: ", relation_embedding.shape)

        if input_data_type == "concept":
            for con, con_embed in zip(batch[0], concept_embedding):
                con_embedding[con] = to_cpu(con_embed)

        elif input_data_type == "property":
            for prop, prop_embed in zip(batch[1], property_embedding):
                prop_embedding[prop] = to_cpu(prop_embed)

        elif input_data_type == "concept_and_property":
            for con, con_embed in zip(batch[0], concept_embedding):
                if con not in con_embedding:
                    con_embedding[con] = to_cpu(con_embed)
                # else:
                # log.info(f"Concept : {con} is already in dictionary !!")

            for prop, prop_embed in zip(batch[1], property_embedding):
                if prop not in prop_embedding:
                    prop_embedding[prop] = to_cpu(prop_embed)
                # else:
                # log.info(f"Property : {prop} is already in dictionary !!")

            get_con_prop_logit = True
            if get_con_prop_logit:
                logits_batch = (
                    torch.flatten(torch.sigmoid(logits)).cpu().numpy().tolist()
                )

                print(f"logits_batch : {logits_batch}", flush=True)

                logits_list.extend(
                    [
                        (con, prop, lgts)
                        for con, prop, lgts in zip(batch[0], batch[1], logits_batch)
                    ]
                )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()

    save_dir = inference_params["save_dir"]

    if input_data_type == "concept":
        file_name = dataset_params["dataset_name"] + "_concept_embeddings.pkl"

        embedding_save_file_name = os.path.join(save_dir, file_name)

        with open(embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(con_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        log.info(f"{'*' * 20} Finished {'*' * 20}")
        log.info("Finished Generating the Concept Embeddings")
        log.info(f"Concept Embeddings are saved in : {embedding_save_file_name}")
        log.info(f"{'*' * 40}")

        return embedding_save_file_name

    if input_data_type == "property":
        file_name = dataset_params["dataset_name"] + "_property_embeddings.pkl"
        embedding_save_file_name = os.path.join(save_dir, file_name)

        with open(embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(prop_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        log.info(f"{'*' * 20} Finished {'*' * 20}")
        log.info("Finished Generating the Property Embeddings")
        log.info(f"Property Embeddings are saved in : {embedding_save_file_name}")
        log.info(f"{'*' * 40}")

        return embedding_save_file_name

    if input_data_type == "concept_and_property":
        con_file_name = dataset_params["dataset_name"] + "_concept_embeddings.pkl"
        prop_file_name = dataset_params["dataset_name"] + "_property_embeddings.pkl"

        logits_file_name = (
            dataset_params["dataset_name"].replace(".tsv", "") + "_logits.tsv"
        )

        con_embedding_save_file_name = os.path.join(save_dir, con_file_name)
        prop_embedding_save_file_name = os.path.join(save_dir, prop_file_name)
        logits_save_file_name = os.path.join(save_dir, logits_file_name)

        with open(con_embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(con_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        with open(prop_embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(prop_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        logit_df = pd.DataFrame.from_records(logits_list)
        print(f"Final logit_df : {logit_df}", flush=True)

        logit_df.to_csv(
            logits_save_file_name,
            sep="\t",
            index=None,
            header=None,
            float_format="%.5f",
        )

        log.info(f"{'*' * 20} Finished {'*' * 20}")
        log.info("Finished Generating the Concept and Property Embeddings and logits")
        log.info(f"Concept Embeddings are saved in : {con_embedding_save_file_name}")
        log.info(f"Property Embeddings are saved in : {prop_embedding_save_file_name}")
        log.info(f"Concept Property Logits are saved in : {logits_save_file_name}")
        log.info(f"{'*' * 40}")

        return con_embedding_save_file_name, prop_embedding_save_file_name


if __name__ == "__main__":

    set_seed(42)

    log.info(f"\n {'*' * 50}")
    log.info(f"Generating the Concept Property Embeddings")

    parser = argparse.ArgumentParser(
        description="Pretrained Concept Property Biencoder Model"
    )

    parser.add_argument(
        "--config_file",
        default="configs/default_config.json",
        help="path to the configuration file",
    )

    args = parser.parse_args()

    log.info(f"Reading Configuration File: {args.config_file}")
    config = read_config(args.config_file)

    log.info("The program is run with following configuration")
    log.info(f"{config} \n")

    inference_params = config.get("inference_params")

    ######################### Important Flags #########################

    get_con_prop_embeds = inference_params["get_con_prop_embeds"]
    get_con_sim_vocab_properties = inference_params["get_con_sim_vocab_properties"]
    get_predict_prop_similar_props = inference_params["get_predict_prop_similar_props"]
    get_con_only_similar_data = inference_params["con_only_similar_data"]

    ######################### Important Flags #########################

    log.info(
        f"Get Concept, Property or Concept and Property Embedings : {get_con_prop_embeds}"
    )
    log.info(f"Get Concept Similar Vocab Properties  : {get_con_sim_vocab_properties} ")
    log.info(
        f"Get Predict Similar JE Filtered Properties  : {get_predict_prop_similar_props} "
    )
    log.info(
        f"Get Concept Only Similar Property Conjuction Data : {get_con_only_similar_data}"
    )

    test_dir = inference_params["test_dir"]
    input_files = os.listdir(test_dir)

    log.info(f"test_dir: {test_dir}")
    log.info(f"input_files: {input_files}")

    for i, inp_file in enumerate(input_files):

        inp_file_path = os.path.join(test_dir, inp_file)

        log.info(f"Processing file: {inp_file}, {i+1} / {len(input_files)}")
        log.info(f"File location: {inp_file_path}")

        inference_params["concept_property_file"] = inp_file_path
        config["dataset_params"]["dataset_name"] = inp_file

        log.info(
            f'Input file in config :{config["inference_params"]["concept_property_file"]}'
        )

        if get_con_prop_embeds:
            input_data_type = inference_params["input_data_type"]

            assert input_data_type in (
                "concept",
                "property",
                "concept_and_property",
            ), "Please specify 'input_data_type' \
                from ('concept', 'property', 'concept_and_property')"

            if input_data_type == "concept":
                concept_pkl_file = generate_embeddings(config=config)

            elif input_data_type == "property":
                property_pkl_file = generate_embeddings(config=config)

            elif input_data_type == "concept_and_property":
                concept_pkl_file, property_pkl_file = generate_embeddings(config=config)
