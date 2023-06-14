import os
import argparse


import logging
import os
import pickle
import pandas as pd
import numpy as np

import torch

import nltk
from nltk.stem import WordNetLemmatizer
from utils.functions import (
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

        logits_file_name = dataset_params["dataset_name"] + "_logits.tsv"

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

    #################
    file_name = os.path.join(save_dir, dataset_params["dataset_name"]) + ".tsv"
    #################

    total_sim_props = 0
    with open(file_name, "w") as file:
        for con_idx, prop_idx in enumerate(con_indices):
            concept = concepts[con_idx]
            similar_properties = [properties[idx] for idx in prop_idx]

            # similar_properties = [
            #     prop
            #     for prop in similar_properties
            #     if not match_multi_words(concept, prop)
            # ]

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


def get_predict_prop_similar_properties(
    input_file,
    con_similar_prop,
    prop_vocab_embed_pkl,
    predict_prop_embed_pkl,
    save_file,
    num_prop_conjuct=5,
):
    # je_filtered_con_prop_file = "siamese_concept_property/data/train_data/joint_encoder_property_conjuction_data/je_filtered_con_similar_vocab_properties.txt"
    # je_filtered_prop_embed_pkl = "/home/amitgajbhiye/cardiff_work/concept_property_embeddings/prop_vocab_500k_mscg_embeds_property_embeddings.pkl"
    # predict_prop_embed_pkl = "/home/amitgajbhiye/cardiff_work/concept_property_embeddings/predict_property_embeds_cnet_premium_property_embeddings.pkl"

    file_name, file_ext = os.path.splitext(input_file)

    print(f"Input File Extension : {file_ext}")
    log.info(f"Input File Extension : {file_ext}")

    if file_ext in (".txt", ".tsv"):
        input_df = pd.read_csv(
            input_file, sep="\t", names=["concept", "property", "label"]
        )
    elif file_ext == ".pkl":
        with open(input_file, "rb") as pkl_file:
            input_df = pickle.load(pkl_file)

    elif isinstance(input_file, pd.DataFrame):
        input_df = input_file
    else:
        print((f"Input File Extension is not correct."))
        log.info(f"Input File Extension is not correct.")

    input_df.rename(
        columns={
            "concept": "concept",
            "property": "predict_property",
            "label": "label",
        },
        inplace=True,
    )

    print("*" * 50)
    print(input_df.head(n=20))
    print("*" * 50)

    log.info("*" * 50)
    log.info(input_df.head(n=20))
    log.info("*" * 50)

    input_concepts = input_df["concept"].unique()
    input_predict_props = input_df["predict_property"].unique()

    num_input_concepts = len(input_concepts)
    num_input_predict_props = len(input_predict_props)

    je_filtered_con_prop_df = pd.read_csv(
        con_similar_prop, sep="\t", names=["concept", "similar_property"]
    )

    je_filtered_concepts = je_filtered_con_prop_df["concept"].unique().tolist()

    ######## Concepts that do not have similar properties ########

    no_similar_prop_concept = set(input_concepts).difference(set(je_filtered_concepts))

    print(
        f"Concepts with no similar properties : {len(no_similar_prop_concept)} , {no_similar_prop_concept}"
    )
    log.info(
        f"Concepts with no similar properties : {len(no_similar_prop_concept)} , {no_similar_prop_concept}"
    )

    with open(prop_vocab_embed_pkl, "rb") as prop_vocab_pkl:
        prop_vocab_embeds_dict = pickle.load(prop_vocab_pkl)

    with open(predict_prop_embed_pkl, "rb") as predict_prop_pkl:
        predict_prop_embeds_dict = pickle.load(predict_prop_pkl)

    print(
        f"JE Filtered Concept Similar Properties DF Shape: {je_filtered_con_prop_df.shape}",
        flush=True,
    )
    print(
        f"Number of Unique Concept in je_filtered_con_prop_df : {len(je_filtered_concepts)}"
    )

    print(
        f"Unique Properties in JE Filtered Con Prop Df : {len(je_filtered_con_prop_df['similar_property'].unique())}",
        flush=True,
    )

    print(
        f"Property Vocab Embeddings : {len(prop_vocab_embeds_dict.keys())}",
        flush=True,
    )

    print()
    print(f"Input DF Shape : {input_df.shape}", flush=True)
    print(f"#Unique input concepts : {num_input_concepts}", flush=True)
    print(f"#Unique input predict properties : {num_input_predict_props}", flush=True)

    all_data, concepts_with_no_similar_props = [], []
    concepts_with_one_similar_prop = 0
    for idx, (concept, predict_property, label) in enumerate(
        zip(input_df["concept"], input_df["predict_property"], input_df["label"])
    ):
        print(f"Processing Concept : {concept}, {idx+1} / {input_df.shape[0]}")
        print(
            f"Concept, Predict Property, Label : {(concept, predict_property, label)}"
        )

        if concept not in set((je_filtered_concepts)):
            concepts_with_no_similar_props.append(concept)
            print(f"Concept : {concept}, has no similar properties")
            conjuct_similar_props = "no_similar_property"
            all_data.append(
                [concept, conjuct_similar_props, predict_property, int(label)]
            )
            continue

        else:
            similar_props = (
                je_filtered_con_prop_df[je_filtered_con_prop_df["concept"] == concept][
                    "similar_property"
                ]
                .unique()
                .tolist()
            )

            similar_props = [
                prop
                for prop in similar_props
                if not match_multi_words(predict_property, prop)
            ]

            print(f"similar_props 0 : {similar_props}")

            if len(similar_props) != 0:
                print(f"similar_props 1 : {similar_props}")

                embed_predict_prop = predict_prop_embeds_dict[predict_property]
                embed_similar_prop = [
                    prop_vocab_embeds_dict[prop] for prop in similar_props
                ]

                zero_embed_predict_prop = np.array(
                    np.insert(embed_predict_prop, 0, float(0))
                ).reshape(1, -1)
                transformed_embed_similar_prop = np.array(transform(embed_similar_prop))

                if len(similar_props) >= num_prop_conjuct:
                    num_nearest_neighbours = num_prop_conjuct
                else:
                    num_nearest_neighbours = len(similar_props)

                predict_prop_similar_props = NearestNeighbors(
                    n_neighbors=num_nearest_neighbours, algorithm="brute"
                ).fit(transformed_embed_similar_prop)

                (
                    similar_prop_distances,
                    similar_prop_indices,
                ) = predict_prop_similar_props.kneighbors(zero_embed_predict_prop)

                if similar_prop_indices.shape[1] != 1:
                    similar_prop_indices = np.squeeze(similar_prop_indices)
                else:
                    concepts_with_one_similar_prop += 1
                    print(f"similar_props : {similar_props}")
                    similar_prop_indices = similar_prop_indices[0]

                similar_properties = [
                    similar_props[idx] for idx in similar_prop_indices
                ]

                conjuct_similar_props = ", ".join(similar_properties)

                print(f"Concept : {concept}", flush=True)
                print(f"Predict Property : {predict_property}", flush=True)
                print(f"Predict Property Similar Properties", flush=True)
                print(similar_properties, flush=True)
                print(f"Conjuct Similar Props", flush=True)
                print(conjuct_similar_props, flush=True)
                print("*" * 30, flush=True)
                print(flush=True)

                all_data.append(
                    [concept, conjuct_similar_props, predict_property, int(label)]
                )
            else:
                concepts_with_no_similar_props.append(concept)
                print(f"Concept : {concept}, has no similar properties")
                conjuct_similar_props = "no_similar_property"
                all_data.append(
                    [concept, conjuct_similar_props, predict_property, int(label)]
                )

    df_all_data = pd.DataFrame.from_records(all_data)
    df_all_data.to_csv(save_file, sep="\t", header=None, index=None)

    print(f"concepts_with_one_similar_prop : {concepts_with_one_similar_prop}")
    print(
        f"Concepts With No Similar Properties: {concepts_with_no_similar_props}",
        flush=True,
    )


def create_con_only_similar_data(
    input_file,
    con_similar_file,
    top_k_sim_props,
    save_file,
):
    if isinstance(input_file, pd.DataFrame):
        inp_df = input_file

    else:
        file_name, file_ext = os.path.splitext(input_file)

        log.info(f"Input File Extension : {file_ext}")

        if file_ext in (".txt", ".tsv"):
            inp_df = pd.read_csv(
                input_file,
                sep="\t",
                names=["concept", "property", "label"],
                dtype={"concept": str, "property": str, "label": int},
            )
        elif file_ext == ".pkl":
            with open(input_file, "rb") as pkl_file:
                inp_df = pickle.load(pkl_file)

        else:
            print((f"Input File Extension is not correct."))
            log.info(f"Input File Extension is not correct.")

    inp_df.rename(
        columns={
            "concept": "concept",
            "property": "predict_property",
            "label": "label",
        },
        inplace=True,
    )

    inp_concepts = inp_df["concept"].unique()
    num_inp_concepts = len(inp_concepts)

    print(flush=True)
    print("*" * 50)
    print(f"Input File : {input_file}")
    print(inp_df.shape)
    print(f"num_inp_concepts : {num_inp_concepts}", flush=True)
    print(inp_df.head(n=20))
    print("*" * 50)
    print(flush=True)

    log.info("*" * 50)
    log.info(inp_df.shape)
    log.info(f"num_inp_concepts : {num_inp_concepts}")
    log.info(inp_df.head(n=20))
    log.info("*" * 50)

    #############################

    con_similar_data = pd.read_csv(
        con_similar_file, sep="\t", names=["concept", "similar_property"]
    )
    con_similar_concepts = con_similar_data["concept"].unique()
    num_con_similar_concepts = len(con_similar_concepts)

    print(flush=True)
    print("*" * 50, flush=True)
    print(f"Concept Similar Property File : {con_similar_file}", flush=True)
    print(con_similar_data, flush=True)
    print(f"num_con_similar_concepts : {num_con_similar_concepts}", flush=True)
    print("*" * 50)

    all_prop_augmented_data, concepts_with_no_similar_props = [], []

    for idx, (concept, predict_property, label) in enumerate(
        zip(inp_df["concept"], inp_df["predict_property"], inp_df["label"])
    ):
        print(f"Processing concept : {idx} / {len(inp_df)}", flush=True)
        print(
            f"concept, predict_property, label : {concept, predict_property, label}",
            flush=True,
        )

        if concept not in set(con_similar_concepts):
            similar_props = "no_similar_property"
            concepts_with_no_similar_props.append(concept)

            print(f"Concept : {concept}, has no similar properties", flush=True)
            print(
                f"Augmented Data : {(concept, similar_props, predict_property, label)}",
                flush=True,
            )
            print(flush=True)

            all_prop_augmented_data.append(
                (concept, similar_props, predict_property, label)
            )

            continue
        else:
            similar_props = (
                con_similar_data[con_similar_data["concept"] == concept][
                    "similar_property"
                ]
                .unique()
                .tolist()
            )

            similar_props = similar_props[0:top_k_sim_props]

            similar_props = ", ".join(similar_props)

            all_prop_augmented_data.append(
                (concept, similar_props, predict_property, label)
            )

            print(
                f"Augmented Data : {(concept, similar_props, predict_property, label)}",
                flush=True,
            )
            print(flush=True)

    df = pd.DataFrame.from_records(all_prop_augmented_data)

    print(f"Data After Augmentation : {df.shape}", flush=True)
    print(df, flush=True)

    df.to_csv(save_file, sep="\t", index=None, header=None)

    print(
        f"num concepts_with_no_similar_props : {len(set(concepts_with_no_similar_props))}"
    )
    print(f"concepts_with_no_similar_props : {set(concepts_with_no_similar_props)}")
    print(f"Only Concept Augmented Data Saved in  {save_file}", flush=True)
    log.info(f"Only Concept Augmented Data Saved in  {save_file}")


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

    if get_predict_prop_similar_props:
        pretrain_data = inference_params["pretrain_data"]
        finetune_data = inference_params["finetune_data"]

        num_prop_conjuct = inference_params["num_prop_conjuct"]

        concept_similar_prop_file = inference_params.get("concept_similar_prop_file")
        vocab_property_embed_pkl = inference_params.get("vocab_property_embed_pkl")
        predict_property_embed_pkl = inference_params.get("predict_property_embed_pkl")

        save_dir = inference_params["save_dir"]

        print(flush=True)
        print(f"pretrain_data : {pretrain_data}", flush=True)
        print(f"finetune_data : {finetune_data}", flush=True)

        print(f"num_prop_conjuct : {num_prop_conjuct}", flush=True)

        print(f"predict_property_embed_pkl : {predict_property_embed_pkl}", flush=True)
        print(f"vocab_property_embed_pkl : {vocab_property_embed_pkl}", flush=True)
        print(f"concept_similar_prop_file : {concept_similar_prop_file}", flush=True)

        print(f"save_dir : {save_dir}", flush=True)
        print(flush=True)

        if pretrain_data:
            train_file_path = inference_params["pretrain_train_file"]
            path, train_filename = os.path.split(train_file_path)

            save_prefix = inference_params["save_prefix"]
            save_train_file_path = os.path.join(
                save_dir, f"{save_prefix}_{train_filename}"
            )

            print(flush=True)
            print(f"Train File Path : {train_file_path}", flush=True)
            print(f"Train Save File Path : {save_train_file_path}", flush=True)
            print(flush=True)

            valid_file_path = inference_params["pretrain_valid_file"]

            path, valid_file_name = os.path.split(valid_file_path)

            save_valid_file_path = os.path.join(
                save_dir, f"{save_prefix}_{valid_file_name}"
            )

            print(flush=True)
            print(f"Valid File Path : {valid_file_path}", flush=True)
            print(f"Valid Save File Path : {save_valid_file_path}", flush=True)
            print(flush=True)

            get_predict_prop_similar_properties(
                input_file=train_file_path,
                con_similar_prop=concept_similar_prop_file,
                prop_vocab_embed_pkl=vocab_property_embed_pkl,
                predict_prop_embed_pkl=predict_property_embed_pkl,
                save_file=save_train_file_path,
                num_prop_conjuct=num_prop_conjuct,
            )

            get_predict_prop_similar_properties(
                input_file=valid_file_path,
                con_similar_prop=concept_similar_prop_file,
                prop_vocab_embed_pkl=vocab_property_embed_pkl,
                predict_prop_embed_pkl=predict_property_embed_pkl,
                save_file=save_valid_file_path,
                num_prop_conjuct=num_prop_conjuct,
            )

        elif finetune_data:
            split_type = inference_params["split_type"]
            log.info(f"Split Type : {split_type}")

            if split_type not in (
                "concept_split",
                "property_split",
                "concept_property_split",
            ):
                raise NameError(
                    "Specify split from : 'concept_split', 'property_split', 'concept_property_split'"
                )

            fold_file_base_path = inference_params["fold_file_base_path"]
            save_prefix = inference_params["save_prefix"]

            if split_type == "property_split":
                num_folds = 5

                train_file_suffix = "train_prop_split_con_prop.pkl"
                test_file_suffix = "test_prop_split_con_prop.pkl"

            elif split_type == "concept_property_split":
                num_folds = 9

                train_file_suffix = "------------"
                test_file_suffix = "------------"
                save_file_suffix = "------------"

            for fold_num in range(num_folds):
                train_file = os.path.join(
                    fold_file_base_path, f"{fold_num}_{train_file_suffix}"
                )
                test_file = os.path.join(
                    fold_file_base_path, f"{fold_num}_{test_file_suffix}"
                )

                with open(train_file, "rb") as train_pkl, open(
                    test_file, "rb"
                ) as test_pkl:
                    train_df = pickle.load(train_pkl)
                    test_df = pickle.load(test_pkl)

                train_save_file_name = os.path.join(
                    save_dir,
                    f"{save_prefix}_{fold_num}_train_prop_conj_{split_type}.tsv",
                )
                test_save_file_name = os.path.join(
                    save_dir,
                    f"{save_prefix}_{fold_num}_test_prop_conj_{split_type}.tsv",
                )

                log.info(f"Fold Number : {fold_num}")
                log.info(f"Train File : {train_file}")
                log.info(f"Test FIle : {test_file}")

                print(f"Fold Number : {fold_num}")
                print(f"Train File : {train_file}")
                print(f"Test FIle : {test_file}")

                get_predict_prop_similar_properties(
                    input_file=train_file,
                    con_similar_prop=concept_similar_prop_file,
                    prop_vocab_embed_pkl=vocab_property_embed_pkl,
                    predict_prop_embed_pkl=predict_property_embed_pkl,
                    save_file=train_save_file_name,
                    num_prop_conjuct=num_prop_conjuct,
                )

                get_predict_prop_similar_properties(
                    input_file=test_file,
                    con_similar_prop=concept_similar_prop_file,
                    prop_vocab_embed_pkl=vocab_property_embed_pkl,
                    predict_prop_embed_pkl=predict_property_embed_pkl,
                    save_file=test_save_file_name,
                    num_prop_conjuct=num_prop_conjuct,
                )

    if get_con_only_similar_data:
        pretrain_data = inference_params["pretrain_data"]
        finetune_data = inference_params["finetune_data"]

        if pretrain_data:
            log.info(f"Pretrain Data")

            train_file = inference_params["pretrain_train_file"]
            valid_file = inference_params["pretrain_valid_file"]

            con_similar_file = inference_params["concept_similar_prop_file"]
            save_prefix = inference_params["save_prefix"]
            save_dir = inference_params["save_dir"]

            top_k_sim_props = inference_params["top_k_sim_props"]

            log.info(f"train_file  : {train_file}")
            log.info(f"valid_file  : {valid_file}")
            log.info(f"con_similar_file  : {con_similar_file}")

            log.info(f"save_prefix  : {save_prefix}")
            log.info(f"save_dir  : {save_dir}")

            train_save_file_name = os.path.join(
                save_dir,
                f"{save_prefix}_pretrain_train_prop_conj.tsv",
            )
            valid_save_file_name = os.path.join(
                save_dir,
                f"{save_prefix}_pretrain_valid_prop_conj.tsv",
            )

            create_con_only_similar_data(
                input_file=train_file,
                con_similar_file=con_similar_file,
                top_k_sim_props=top_k_sim_props,
                save_file=train_save_file_name,
            )

            create_con_only_similar_data(
                input_file=valid_file,
                con_similar_file=con_similar_file,
                top_k_sim_props=top_k_sim_props,
                save_file=valid_save_file_name,
            )

        elif finetune_data:
            split_type = inference_params["split_type"]
            log.info(f"Split Type : {split_type}")

            if split_type not in (
                "concept_split",
                "property_split",
                "concept_property_split",
            ):
                raise NameError(
                    "Specify split from : 'concept_split', 'property_split', 'concept_property_split'"
                )

            con_similar_file = inference_params["concept_similar_prop_file"]
            save_prefix = inference_params["save_prefix"]
            save_dir = inference_params["save_dir"]
            top_k_sim_props = inference_params["top_k_sim_props"]

            fold_file_base_path = inference_params["fold_file_base_path"]

            log.info(f"con_similar_file : {con_similar_file}")
            log.info(f"save_dir : {save_dir}")
            log.info(f"save_prefix : {save_prefix}")
            log.info(f"top_k_sim_props : {top_k_sim_props}")
            log.info(f"fold_file_base_path : {fold_file_base_path}")

            if split_type == "property_split":
                num_folds = 5

                train_file_suffix = "train_prop_split_con_prop.pkl"
                test_file_suffix = "test_prop_split_con_prop.pkl"

            elif split_type == "concept_property_split":
                num_folds = 9

                train_file_suffix = "------------"
                test_file_suffix = "------------"
                save_file_suffix = "------------"

            for fold_num in range(num_folds):
                train_file = os.path.join(
                    fold_file_base_path, f"{fold_num}_{train_file_suffix}"
                )
                test_file = os.path.join(
                    fold_file_base_path, f"{fold_num}_{test_file_suffix}"
                )

                with open(train_file, "rb") as train_pkl, open(
                    test_file, "rb"
                ) as test_pkl:
                    train_df = pickle.load(train_pkl)
                    test_df = pickle.load(test_pkl)

                train_save_file_name = os.path.join(
                    save_dir,
                    f"{save_prefix}_{fold_num}_train_prop_conj_{split_type}.tsv",
                )
                test_save_file_name = os.path.join(
                    save_dir,
                    f"{save_prefix}_{fold_num}_test_prop_conj_{split_type}.tsv",
                )

                log.info(f"Fold Number : {fold_num}")
                log.info(f"Train File : {train_file}")
                log.info(f"Test FIle : {test_file}")

                print(f"Fold Number : {fold_num}")
                print(f"Train File : {train_file}")
                print(f"Test FIle : {test_file}")

                create_con_only_similar_data(
                    input_file=train_df,
                    con_similar_file=con_similar_file,
                    top_k_sim_props=top_k_sim_props,
                    save_file=train_save_file_name,
                )

                create_con_only_similar_data(
                    input_file=test_df,
                    con_similar_file=con_similar_file,
                    top_k_sim_props=top_k_sim_props,
                    save_file=test_save_file_name,
                )
