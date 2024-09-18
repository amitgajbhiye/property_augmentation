import argparse
import logging
import os
import pickle

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import NearestNeighbors

from utils.functions import (
    create_model,
    mcrae_dataset_and_dataloader,
    read_config,
    to_cpu,
)

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

        data_df = data_df[["concept", "property", "label"]]

        log.info(f"Final Df")
        log.info(data_df.head(n=100))

        return data_df

    elif input_data_type == "property" and num_columns == 1:
        log.info("Generating Embeddings for Properties")
        log.info(f"Number of records : {data_df.shape[0]}")

        data_df.rename(columns={0: "property"}, inplace=True)

        unique_properties = data_df["property"].unique()
        data_df = pd.DataFrame(unique_properties, columns=["property"])

        data_df["concept"] = "dummy_concept"
        data_df["label"] = int(0)

        data_df = data_df[["concept", "property", "label"]]
        log.info(f"Final Df")
        log.info(data_df.head(n=100))

        return data_df

    elif input_data_type == "concept_and_property" and num_columns in (2, 4):
        log.info("Generating Embeddings for Concepts and Properties")
        log.info(f"Number of records : {data_df.shape[0]}")

        log.info(f"Input Df - get_property_similar_concepts module")
        log.info(data_df.head(n=100))
        data_df["label"] = int(0)

        df3col = data_df.rename(columns={0: "concept", 1: "property"}, inplace=False)

        df3col = df3col[["concept", "property", "label"]]

        log.info(f"Final Df")
        log.info(df3col.head(n=100))

        return df3col

    else:
        raise Exception(
            f"Please Enter a Valid Input data type from: 'concept', 'property' or conncept_and_property. \
            Current 'input_data_type' is: {input_data_type}"
        )


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

    # con_prop_embed_save_prefix = inference_params["con_prop_embed_save_prefix"]
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

        con_embedding_save_file_name = os.path.join(save_dir, con_file_name)
        prop_embedding_save_file_name = os.path.join(save_dir, prop_file_name)

        with open(con_embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(con_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        with open(prop_embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(prop_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        log.info(f"{'*' * 20} Finished {'*' * 20}")
        log.info("Finished Generating the Concept and Property Embeddings")
        log.info(f"Concept Embeddings are saved in : {con_embedding_save_file_name}")
        log.info(f"Property Embeddings are saved in : {prop_embedding_save_file_name}")
        log.info(f"{'*' * 40}")

        return con_embedding_save_file_name, prop_embedding_save_file_name


######################################
def transform(vecs):
    maxnorm = max([np.linalg.norm(v) for v in vecs])
    new_vecs = []

    for v in vecs:
        new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm**2 - np.linalg.norm(v) ** 2)))

    return new_vecs


def match_multi_words(word1, word2):
    lemmatizer = WordNetLemmatizer()

    word1 = " ".join([lemmatizer.lemmatize(word) for word in word1.split()])
    word2 = " ".join([lemmatizer.lemmatize(word) for word in word2.split()])

    return word1 == word2


def get_property_similar_concepts(
    config,
    concept_embed_pkl,
    vocab_property_embed_pkl,
):
    log.info(f"Getting Property Similar Concepts ....")
    log.info(f"concept_embed_pkl : {concept_embed_pkl}")
    log.info(f"vocab_property_embed_pkl : {vocab_property_embed_pkl}")

    inference_params = config.get("inference_params")

    input_data_type = inference_params["input_data_type"]
    log.info(f"Input Data Type : {input_data_type}")

    dataset_params = config.get("dataset_params")
    save_dir = inference_params["save_dir"]

    with open(concept_embed_pkl, "rb") as con_pkl_file, open(
        vocab_property_embed_pkl, "rb"
    ) as prop_pkl_file:
        con_dict = pickle.load(con_pkl_file)
        prop_dict = pickle.load(prop_pkl_file)

    concepts = list(con_dict.keys())
    con_embeds = list(con_dict.values())

    zero_con_embeds = np.array([np.insert(l, 0, float(0)) for l in con_embeds])
    transformed_con_embeds = np.array(transform(con_embeds))

    log.info(f"******* In get_property_similar_concepts function *******")
    log.info(f"******* Input Concept Embedding Details **********")
    log.info(f"Number of Concepts : {len(concepts)}")
    log.info(f"Length of Concepts Embeddings : {len(con_embeds)}")
    log.info(f"Shape of zero_con_embeds: {zero_con_embeds.shape}")
    log.info(f"Shape of transformed_con_embeds : {transformed_con_embeds.shape}")

    properties = list(prop_dict.keys())
    prop_embeds = list(prop_dict.values())

    zero_prop_embeds = np.array([np.insert(l, 0, 0) for l in prop_embeds])
    transformed_prop_embeds = np.array(transform(prop_embeds))

    log.info(f"******* Vocab Property Embedding Details **********")
    log.info(f"Number of Vocab Properties : {len(properties)}")
    log.info(f"Length of Vocab Properties Embeddings : {len(prop_embeds)}")
    log.info(f"Shape of zero_prop_embeds: {zero_prop_embeds.shape}")
    log.info(f"Shape of transformed_prop_embeds : {transformed_prop_embeds.shape}")

    # Learning Nearest Neighbours
    # num_nearest_neighbours = 5

    num_nearest_neighbours = inference_params["num_nearest_neighbours"]

    print(f"Learning {num_nearest_neighbours} neighbours !!")

    property_similar_concepts = NearestNeighbors(
        n_neighbors=num_nearest_neighbours, algorithm="brute"
    ).fit(np.array(transformed_con_embeds))

    prop_distances, prop_indices = property_similar_concepts.kneighbors(
        np.array(zero_prop_embeds)
    )
    log.info(f"num_concepts : {concepts}")
    log.info(f"num_properties : {properties}")

    log.info(f"prop_distances : {prop_distances.shape}")
    log.info(f"prop_indices : {prop_indices.shape}")

    # file_name = os.path.join(save_dir, "test_property_similar_concepts") + ".tsv"

    #########################################
    prop_similar_concepts_save_file_name = os.path.join(
        save_dir, f"{dataset_params['dataset_name']}.tsv"
    )
    #########################################

    total_sim_cons = 0

    with open(prop_similar_concepts_save_file_name, "w") as file:
        for prop_idx, con_idx in enumerate(prop_indices):
            prop = properties[prop_idx]
            similar_concepts = [concepts[idx] for idx in con_idx]

            print(f"num_similar_concepts : {len(similar_concepts)}", flush=True)
            print(f"prop:sim_concepts-\t{prop}:{similar_concepts}")

            con_prop_list = [(sim_con, prop) for sim_con in similar_concepts]
            print(f"con_prop_list : {con_prop_list}\n")

            total_sim_cons += len(similar_concepts)

            for sim_con in similar_concepts:
                line = sim_con + "\t" + prop + "\n"
                file.write(line)

    log.info(f"Total Number of input concepts : {len(concepts)}")
    log.info(f"Total Sim Properties Generated : {total_sim_cons}")
    log.info(f"Finished getting similar properties")
    log.info(
        f"properties_similar_to_concepts_file: {prop_similar_concepts_save_file_name}"
    )


def get_concept_similar_vocab_properties(
    config, concept_embed_pkl, vocab_property_embed_pkl
):
    log.info(f"Getting Concept Similar Vocab Properties ....")

    inference_params = config.get("inference_params")
    # input_data_type = inference_params["input_data_type"]
    # log.info(f"Input Data Type : {input_data_type}")

    dataset_params = config.get("dataset_params")
    save_dir = inference_params["save_dir"]

    with open(concept_embed_pkl, "rb") as con_pkl_file, open(
        vocab_property_embed_pkl, "rb"
    ) as prop_pkl_file:
        con_dict = pickle.load(con_pkl_file)
        prop_dict = pickle.load(prop_pkl_file)

    concepts = list(con_dict.keys())
    con_embeds = list(con_dict.values())

    zero_con_embeds = np.array([np.insert(l, 0, float(0)) for l in con_embeds])
    transformed_con_embeds = np.array(transform(con_embeds))

    log.info(f"******* In get_concept_similar_vocab_properties function *******")
    log.info(f"******* Input Concept Embedding Details **********")
    log.info(f"Number of Concepts : {len(concepts)}")
    log.info(f"Length of Concepts Embeddings : {len(con_embeds)}")
    log.info(f"Shape of zero_con_embeds: {zero_con_embeds.shape}")
    log.info(f"Shape of transformed_con_embeds : {transformed_con_embeds.shape}")

    properties = list(prop_dict.keys())
    prop_embeds = list(prop_dict.values())

    zero_prop_embeds = np.array([np.insert(l, 0, 0) for l in prop_embeds])
    transformed_prop_embeds = np.array(transform(prop_embeds))

    log.info(f"******* Vocab Property Embedding Details **********")
    log.info(f"Number of Vocab Properties : {len(properties)}")
    log.info(f"Length of Vocab Properties Embeddings : {len(prop_embeds)}")
    log.info(f"Shape of zero_prop_embeds: {zero_prop_embeds.shape}")
    log.info(f"Shape of transformed_prop_embeds : {transformed_prop_embeds.shape}")

    prop_dict_transform = {
        prop: trans for prop, trans in zip(properties, transformed_prop_embeds)
    }
    prop_dict_zero = {prop: trans for prop, trans in zip(properties, zero_prop_embeds)}

    # Learning Nearest Neighbours
    # num_nearest_neighbours = 50
    num_nearest_neighbours = inference_params["num_nearest_neighbours"]

    log.info(f"Learning {num_nearest_neighbours} neighbours !!")

    con_similar_properties = NearestNeighbors(
        n_neighbors=num_nearest_neighbours, algorithm="brute"
    ).fit(np.array(transformed_prop_embeds))

    con_distances, con_indices = con_similar_properties.kneighbors(
        np.array(zero_con_embeds)
    )

    log.info(f"con_distances shape : {con_distances.shape}")
    log.info(f"con_indices shape : {con_indices.shape}")

    con_similar_prop_dict = {}
    file_name = os.path.join(save_dir, dataset_params["dataset_name"]) + ".tsv"

    total_sim_props = 0
    with open(file_name, "w") as file:
        for con_idx, prop_idx in enumerate(con_indices):
            concept = concepts[con_idx]
            similar_properties = [properties[idx] for idx in prop_idx]

            similar_properties = [
                prop
                for prop in similar_properties
                if not match_multi_words(concept, prop)
            ]

            con_similar_prop_dict[concept] = similar_properties

            print(f"Number Similar Props : {len(similar_properties)}")
            print(f"{concept} \t {similar_properties}\n")

            total_sim_props += len(similar_properties)

            for prop in similar_properties:
                line = concept + "\t" + prop + "\n"
                file.write(line)

    log.info(f"Total Number of input concepts : {len(concepts)}")
    log.info(f"Total Sim Properties Generated : {total_sim_props}")
    log.info(f"Finished getting similar properties")


if __name__ == "__main__":
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
    # get_predict_prop_similar_props = inference_params["get_predict_prop_similar_props"]
    # get_con_only_similar_data = inference_params["con_only_similar_data"]
    get_props_sim_concepts = inference_params["get_props_sim_concepts"]

    ######################### Important Flags #########################

    log.info("*" * 60)
    log.info(
        f"Get Concept, Property or Concept and Property Embedings : {get_con_prop_embeds}"
    )
    log.info(f"Get Concept Similar Vocab Properties  : {get_con_sim_vocab_properties} ")

    log.info(f"Get Property Similar Concepts : {get_props_sim_concepts}")

    log.info("*" * 60)

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

    if get_con_sim_vocab_properties:
        concept_embed_pkl = inference_params["concept_embed_pkl"]
        vocab_property_embed_pkl = inference_params["vocab_property_embed_pkl"]

        get_concept_similar_vocab_properties(
            config,
            concept_embed_pkl=concept_embed_pkl,
            vocab_property_embed_pkl=vocab_property_embed_pkl,
        )

    if get_props_sim_concepts:
        log.info(f"in_flag : {get_props_sim_concepts}")

        # Temporary for McRae Analysis Paper 1
        # concept_pkl_file = "trained_models/mcrae_analysis_exp/prop_similar_analysis/mcrae_train_test_concept_embeddings.pkl"
        # property_pkl_file = "trained_models/mcrae_analysis_exp/prop_similar_analysis/mcrae_train_test_property_embeddings.pkl"

        get_property_similar_concepts(
            config=config,
            concept_embed_pkl=concept_pkl_file,
            vocab_property_embed_pkl=property_pkl_file,
        )
