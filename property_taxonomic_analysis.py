import os

import logging
import os
import pickle
import pandas as pd
import numpy as np

import torch


from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)

cuda_available = torch.cuda.is_available()

device = torch.device("cuda") if cuda_available else torch.device("cpu")


def transform(vecs):
    maxnorm = max([np.linalg.norm(v) for v in vecs])
    new_vecs = []

    for v in vecs:
        new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm**2 - np.linalg.norm(v) ** 2)))

    return new_vecs


def get_nearest_neighbours(embedding_file, output_file, num_nearest_neighbours):
    log.info(f"Getting Nearest Neighbors ....")

    print(f"input_embedding_file_name : {embedding_file}", flush=True)
    print(f"input_embedding_file_name : {embedding_file}", flush=True)

    with open(embedding_file, "rb") as con_pkl_file:
        emb_dict = pickle.load(con_pkl_file)

    concepts = list(emb_dict.keys())
    con_embeds = list(emb_dict.values())

    zero_con_embeds = np.array([np.insert(l, 0, float(0)) for l in con_embeds])
    transformed_con_embeds = np.array(transform(con_embeds))

    log.info(f"******* In get_concept_similar_vocab_properties function *******")
    log.info(f"******* Input Concept Embedding Details **********")
    log.info(f"Number of Concepts : {len(concepts)}")
    log.info(f"Length of Concepts Embeddings : {len(con_embeds)}")
    log.info(f"Shape of zero_con_embeds: {zero_con_embeds.shape}")
    log.info(f"Shape of transformed_con_embeds : {transformed_con_embeds.shape}")

    log.info(f"Learning {num_nearest_neighbours} neighbours !!")

    con_similar_cons = NearestNeighbors(
        n_neighbors=num_nearest_neighbours, algorithm="brute"
    ).fit(np.array(transformed_con_embeds))

    con_distances, con_indices = con_similar_cons.kneighbors(np.array(zero_con_embeds))

    log.info(f"con_distances shape : {con_distances.shape}")
    log.info(f"con_indices shape : {con_indices.shape}")

    total_sim_cons = 0
    with open(output_file, "w") as file:
        for con_idx, similar_idx in enumerate(con_indices):
            concept = concepts[con_idx]
            similar_concepts = [concepts[idx] for idx in similar_idx]

            line = f"{concept}\t{','.join(similar_concepts)}"

            print(f"Number Similar Concepts : {len(similar_concepts)}", flush=True)
            print(line, flush=True)
            print(flush=True)
            file.write(f"{line}\n")

            total_sim_cons += len(similar_concepts)

    log.info(f"Total Number of input concepts : {len(concepts)}")
    log.info(f"Total Sim Concepts Generated : {total_sim_cons}")
    log.info(f"Finished getting similar concepts")


if __name__ == "__main__":
    embedding_file = (
        "trained_models/taxonomic_analysis/ufet_type_concept_embeddings.pkl"
    )
    output_file = "trained_models/taxonomic_analysis/ufet_type_similar_50_types.tsv"
    num_nearest_neighbours = 50

    get_nearest_neighbours(embedding_file, output_file, num_nearest_neighbours)
