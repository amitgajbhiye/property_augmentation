import pickle

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import NearestNeighbors


def match_props(predict_prop, similar_prop):

    lemmatizer = WordNetLemmatizer()

    predict_prop = " ".join(
        [lemmatizer.lemmatize(word) for word in predict_prop.split()]
    )
    similar_prop = " ".join(
        [lemmatizer.lemmatize(word) for word in similar_prop.split()]
    )

    return predict_prop == similar_prop


def transform(vecs):

    maxnorm = max([np.linalg.norm(v) for v in vecs])
    new_vecs = []

    for v in vecs:
        new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm ** 2 - np.linalg.norm(v) ** 2)))

    return new_vecs


def new_random_and_similar_conjuct_properties(input_file, save_file):

    # Input will be cnet premium train and validation file.

    je_filtered_con_prop_file = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/je_filtered_con_similar_vocab_properties.txt"
    je_filtered_prop_embed_pkl = "/scratch/c.scmag3/biencoder_concept_property/trained_models/con_pro_embeddings/prop_vocab_500k_mscg_embeds_property_embeddings.pkl"

    je_filtered_con_prop_df = pd.read_csv(
        je_filtered_con_prop_file, sep="\t", names=["concept", "similar_property"]
    )

    with open(je_filtered_prop_embed_pkl, "rb") as je_filter_pkl:
        je_filtered_similar_props_embed = pickle.load(je_filter_pkl)

    input_df = pd.read_csv(input_file, sep="\t", names=["concept", "predict_property"])
    input_concepts = input_df["concept"].unique()

    predict_prop_embed_pkl = "/scratch/c.scmag3/biencoder_concept_property/trained_models/con_pro_embeddings/predict_property_embeds_cnet_premium_property_embeddings.pkl"

    with open(predict_prop_embed_pkl, "rb") as predict_prop:
        predict_prop_embeds = pickle.load(predict_prop)

    print(
        f"JE Filtered Similar Properties DF Shape: {je_filtered_con_prop_df.shape}",
        flush=True,
    )
    print(
        f"Filtered Property Embeddings : {len(je_filtered_similar_props_embed.keys())}",
        flush=True,
    )
    print(
        f"Unique Properties in JE Filtered Con Prop Df : {len(je_filtered_con_prop_df['similar_property'].unique())}",
        flush=True,
    )
    print(flush=True)
    print(f"Input DF Shape : {input_df.shape}", flush=True)
    print(f"#Unique input concepts : {len(input_concepts)}", flush=True)
    print(
        f"Predict Property Embeddings : {len(predict_prop_embeds.keys())}", flush=True
    )
    print(
        f"Unique Properties in Input Df : {len(input_df['predict_property'].unique())}",
        flush=True,
    )

    #### For the train/valid input concept, finding the similar properties from  JE filetered properties
    all_data = []
    for idx, concept in enumerate(input_concepts):

        print(flush=True)
        print(f"Processing Concept {idx+1} / {len(input_concepts)}", flush=True)

        predict_props = (
            input_df[input_df["concept"] == concept]["predict_property"]
            .unique()
            .tolist()
        )

        concept_data = []
        for predict_prop in predict_props:

            similar_props = (
                je_filtered_con_prop_df[je_filtered_con_prop_df["concept"] == concept][
                    "similar_property"
                ]
                .unique()
                .tolist()
            )
            similar_props = [
                prop for prop in similar_props if not match_props(predict_prop, prop)
            ]

            embed_predict_prop = predict_prop_embeds[predict_prop]
            embed_similar_prop = [
                je_filtered_similar_props_embed[prop] for prop in similar_props
            ]

            zero_embed_predict_prop = np.array(
                np.insert(embed_predict_prop, 0, float(0))
            ).reshape(1, -1)
            transformed_embed_similar_prop = np.array(transform(embed_similar_prop))

            if len(similar_props) >= 5:
                num_nearest_neighbours = 5
            else:
                num_nearest_neighbours = len(similar_props)

            predict_prop_similar_props = NearestNeighbors(
                n_neighbors=num_nearest_neighbours, algorithm="brute"
            ).fit(transformed_embed_similar_prop)

            (
                similar_prop_distances,
                similar_prop_indices,
            ) = predict_prop_similar_props.kneighbors(zero_embed_predict_prop)

            similar_prop_indices = np.squeeze(similar_prop_indices)
            similar_properties = [similar_props[idx] for idx in similar_prop_indices]

            # if len(similar_properties) >= 5:
            #     conjuct_similar_props = similar_properties[:5]
            # else:
            #     conjuct_similar_props = similar_properties

            conjuct_similar_props = ", ".join(similar_properties)

            print(f"Concept : {concept}", flush=True)
            print(f"Predict Property : {predict_prop}", flush=True)
            print(f"Predict Property Similar Properties", flush=True)
            print(similar_properties, flush=True)
            print(f"Conjuct Similar Props", flush=True)
            print(conjuct_similar_props, flush=True)
            print("*" * 30, flush=True)
            print(flush=True)

            concept_data.append([concept, conjuct_similar_props, predict_prop])
            all_data.append([concept, conjuct_similar_props, predict_prop])

    # print(f"all_data")
    # print (all_data)

    df_all_data = pd.DataFrame.from_records(all_data)

    df_all_data.to_csv(save_file, sep="\t", header=None, index=None)

    return df_all_data


print(f"************** Train Data Property Conjuction **************", flush=True)
train_file = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_concept_property_data/original_train_gkbcnet_plus_cnethasproperty.tsv"
train_save_file = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/prop_conj_train_cnet_premium.tsv"

prop_conj_train_df = new_random_and_similar_conjuct_properties(
    input_file=train_file, save_file=train_save_file
)

print(flush=True)
print(f"************** Valid Data Property Conjuction **************", flush=True)
valid_file = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_concept_property_data/original_valid_gkbcnet_plus_cnethasproperty.tsv"
valid_save_file = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/prop_conj_valid_cnet_premium.tsv"

prop_conj_valid_df = new_random_and_similar_conjuct_properties(
    input_file=valid_file, save_file=valid_save_file
)

print(f"Train Data Shape : {prop_conj_train_df.shape}", flush=True)
print(f"Valid Data Shape : {prop_conj_valid_df.shape}", flush=True)

# def random_conjuct_properties(input_df, num_random_prop_to_conjuct=None):

#     unique_concepts = input_df["concept"].unique()
#     num_unique_concepts = len(unique_concepts)

#     print(f"Number of Unique Concepts : {num_unique_concepts}")

#     all_random_data = []

#     counter = 0

#     for i, concept in enumerate(unique_concepts):

#         print()
#         print(f"Processing Concept : {i+1}/{num_unique_concepts}, {concept}")

#         concept_data = input_df[input_df["concept"] == concept]
#         properties_for_concept = concept_data["property"].unique()
#         num_properties_for_concept = len(properties_for_concept)

#         print(f"Concept Properties : {properties_for_concept}")
#         print(f"Num Properties for Concept : {num_properties_for_concept}")
#         print()

#         if len(properties_for_concept) < 4:

#             print(f"Case-1")
#             print(
#                 f"***** Properties for the concept is less than 3, {num_properties_for_concept}, Not Enough *****"
#             )
#             print(f"Moving to Next Concept")

#             for predict_prop in properties_for_concept:

#                 print([concept, "Nothing to Conjuct", predict_prop])
#                 all_random_data.append([concept, "Nothing to Conjuct", predict_prop])

#             continue
#         else:
#             for predict_prop in properties_for_concept:

#                 rest_of_prop = [
#                     prop for prop in properties_for_concept if prop != predict_prop
#                 ]
#                 num_rest_of_prop = len(rest_of_prop)

#                 print(f"Predict Property : {predict_prop}")
#                 print(f"Rest of Properties : {rest_of_prop}")
#                 print(f"num_rest_of_prop : {num_rest_of_prop}")

#                 if 3 <= num_rest_of_prop <= 10:

#                     num_prop = random.randint(3, num_rest_of_prop)
#                     random_props = random.sample(rest_of_prop, num_prop)
#                     conjuction_properties = ", ".join(random_props)

#                     all_random_data.append(
#                         [concept, conjuction_properties, predict_prop]
#                     )

#                     print(f"Case-2 : num_rest_of_prop : {num_rest_of_prop}")
#                     print(f"Properties to Conjuct : {conjuction_properties}")
#                     print()

#                 elif num_rest_of_prop >= 10:

#                     num_prop = random.randint(3, 10)
#                     random_props = random.sample(rest_of_prop, num_prop)
#                     conjuction_properties = ", ".join(random_props)

#                     all_random_data.append(
#                         [concept, conjuction_properties, predict_prop]
#                     )
#                     print(f"Case-3 : num_rest_of_prop : {num_rest_of_prop}")
#                     print(f"Properties to Conjuct : {conjuction_properties}")
#                     print()

#                 else:
#                     print(f"Case-4 : None of the above")
#                     print(f"num_rest_of_prop : {num_rest_of_prop}")
#                     print(f"Concept : {concept}")
#                     counter += 1

#     # print (f"All Randomly Generated Properties to Conjuct")
#     # print (all_random_data)

#     random_data_df = pd.DataFrame.from_records(all_random_data)

#     input_unique = set(unique_concepts)
#     generated_unique = set(random_data_df[0].unique())
#     concepts_unique_to_input = input_unique.difference(generated_unique)

#     print()
#     print(f"random_data_df - Unique Concepts : {len(generated_unique)}")
#     print(f"input_df - Unique Concepts -  : {num_unique_concepts}")

#     print(f"Counter for case 4 : {counter}")
#     print(f"all_random_data - len : {len(all_random_data)}")
#     print(f"random_data_df.shape : {random_data_df.shape}")
#     print(f"input_df.shape: {input_df.shape}")

#     print(f"Concepts Unique to Input : {len(concepts_unique_to_input)}")
#     print(concepts_unique_to_input)
#     print(f"Concepts Unique to Random Generated Data")
#     print(generated_unique.difference(input_unique))

#     assert (
#         input_unique == generated_unique
#     ), "Number of Concepts in random_data_df is not equal to input_df"
#     print("Assert in random_conjuct_properties function passed")

#     # file_name = "siamese_concept_property/data/train_data/joint_encoder_property_conjuction_data/random_conjuct_property.tsv"

#     file_name = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/random_conjuct_property.tsv"
#     random_data_df.to_csv(file_name, index=None, header=None, sep="\t")


# def match_props(predict_prop, similar_prop):

#     lemmatizer = WordNetLemmatizer()

#     predict_prop = " ".join(
#         [lemmatizer.lemmatize(word) for word in predict_prop.split()]
#     )
#     similar_prop = " ".join(
#         [lemmatizer.lemmatize(word) for word in similar_prop.split()]
#     )

#     return predict_prop == similar_prop


# def transform(vecs):

#     maxnorm = max([np.linalg.norm(v) for v in vecs])
#     new_vecs = []

#     for v in vecs:
#         new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm ** 2 - np.linalg.norm(v) ** 2)))

#     return new_vecs


# def get_predict_property_similar_properties(
#     predict_prop_embed_pkl_file, property_vocab_embed_pkl_file
# ):

#     """Function to get properties similar to predict property from the property vocab"""

#     with open(predict_prop_embed_pkl_file, "rb") as predict_prop_pkl_file, open(
#         property_vocab_embed_pkl_file, "rb"
#     ) as prop_vocab_pkl_file:

#         predict_prop_dict = pickle.load(predict_prop_pkl_file)
#         prop_vocab_dict = pickle.load(prop_vocab_pkl_file)

#     predict_props = list(predict_prop_dict.keys())
#     predict_props_embeds = list(predict_prop_dict.values())

#     zero_predict_prop_embeds = np.array(
#         [np.insert(l, 0, float(0)) for l in predict_props_embeds]
#     )
#     transformed_predict_prop_embeds = np.array(transform(predict_props_embeds))

#     print(f"In get_predict_property_similar_properties function")
#     print(f"Number of Predict Properties : {len(predict_props)}")
#     print(f"Length of Predict Property Embeddings : {len(predict_props_embeds)}")
#     print(f"Shape of zero_predict_prop_embeds: {zero_predict_prop_embeds.shape}")
#     print(f"Shape of transformed_con_embeds : {transformed_predict_prop_embeds.shape}")

#     properties_vocab = list(prop_vocab_dict.keys())
#     prop_vocab_embeds = list(prop_vocab_dict.values())
#     zero_prop_vocab_embeds = np.array([np.insert(l, 0, 0) for l in prop_vocab_embeds])
#     transformed_prop_vocab_embeds = np.array(transform(prop_vocab_embeds))

#     print(f"Number of Properties in Vocab : {len(properties_vocab)}")
#     print(f"Length of Properties Embeddings : {len(prop_vocab_embeds)}")
#     print(f"Shape of zero_prop_vocab_embeds: {zero_prop_vocab_embeds.shape}")
#     print(
#         f"Shape of transformed_prop_vocab_embeds : {transformed_prop_vocab_embeds.shape}"
#     )

#     prop_vocab_dict_transform = {
#         prop: trans
#         for prop, trans in zip(properties_vocab, transformed_prop_vocab_embeds)
#     }
#     prop_vocab_dict_zero = {
#         prop: trans for prop, trans in zip(properties_vocab, zero_prop_vocab_embeds)
#     }

#     # Learning Nearest Neighbours
#     num_nearest_neighbours = 10

#     predict_prop_similar_vocab_properties = NearestNeighbors(
#         n_neighbors=num_nearest_neighbours, algorithm="brute"
#     ).fit(np.array(transformed_prop_vocab_embeds))

#     (
#         predict_prop_distances,
#         predict_prop_indices,
#     ) = predict_prop_similar_vocab_properties.kneighbors(
#         np.array(zero_predict_prop_embeds)
#     )

#     print(f"predict_prop_distances shape : {predict_prop_distances.shape}")
#     print(f"predict_prop_indices shape : {predict_prop_indices.shape}")

#     predict_prop_similar_vocab_props_dict = {}
#     # file_name = os.path.join(save_dir, dataset_params["dataset_name"]) + ".tsv"

#     file_name = "predict_property_similar_20vocab_properties.tsv"

#     all_similar_data = []

#     for predict_prop_idx, vocab_prop_idx in enumerate(predict_prop_indices):

#         predict_prop = predict_props[predict_prop_idx].replace(".", "")
#         similar_vocab_properties = [properties_vocab[idx] for idx in vocab_prop_idx]

#         # print(f"{predict_prop} \t {similar_vocab_properties}")

#         similar_vocab_properties = [
#             prop
#             for prop in similar_vocab_properties
#             if not match_props(predict_prop, prop)
#         ]

#         num_prop = random.randint(3, 10)

#         similar_vocab_properties = similar_vocab_properties[0:num_prop]
#         predict_prop_similar_vocab_props_dict[predict_prop] = similar_vocab_properties

#         print(f"Select : {num_prop} properties")
#         print(f"{predict_prop}\t{','.join(similar_vocab_properties)}\n")

#         all_similar_data.append([predict_prop, ",".join(similar_vocab_properties)])

#     similar_data_df = pd.DataFrame.from_records(all_similar_data)

#     # file_name = "siamese_concept_property/data/train_data/joint_encoder_property_conjuction_data/predict_property_similar_vocab_properties.tsv"

#     file_name = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/predict_property_similar_vocab_properties.tsv"

#     similar_data_df.to_csv(
#         file_name, index=None, header=None, sep="\t",
#     )

#     print(f"Finished getting similar properties")


# ############## Work TO DO  ##############


# def random_and_similar_conjuct_properties(
#     input_df, similar_prop_file, random_prop_file
# ):

#     random_prop_df = pd.read_csv(
#         random_prop_file,
#         header=None,
#         sep="\t",
#         names=["concept", "random_props", "predict_prop"],
#     )
#     similar_prop_df = pd.read_csv(
#         similar_prop_file,
#         header=None,
#         sep="\t",
#         names=["predict_prop", "similar_props"],
#     )

#     #     print (f"random_prop_df.shape : {random_prop_df.shape}")
#     #     print (random_prop_df.head(n=10))

#     #     print ()
#     #     print (f"similar_prop_df.shape : {similar_prop_df.shape}")
#     #     print (similar_prop_df.head(n=10))

#     unique_concepts = input_df["concept"].unique()
#     num_unique_concepts = len(unique_concepts)

#     all_random_and_similar = []

#     print(f"Number of Unique Concepts : {num_unique_concepts}")

#     for i, concept in enumerate(unique_concepts):

#         print()
#         print(f"Processing Concept : {i+1}/{num_unique_concepts}, {concept}")

#         concept_data = input_df[input_df["concept"] == concept]
#         properties_for_concept = concept_data["property"].unique()
#         num_properties_for_concept = len(properties_for_concept)

#         print(f"Concept Properties : {properties_for_concept}")
#         print(f"Num Properties for Concept : {num_properties_for_concept}")
#         print()

#         for prop in properties_for_concept:

#             print("*************")
#             print(f"Concept: {concept} , Predict Prop : {prop}")
#             print("*************")

#             rand_props = (
#                 random_prop_df[
#                     (random_prop_df["concept"] == concept)
#                     & (random_prop_df["predict_prop"] == prop)
#                 ]["random_props"]
#                 .to_list()[0]
#                 .split(",")
#             )
#             similar_props = (
#                 similar_prop_df[similar_prop_df["predict_prop"] == prop][
#                     "similar_props"
#                 ]
#                 .to_list()[0]
#                 .split(",")
#             )

#             print(f"rand_props : {rand_props}, {len(rand_props)}")
#             print(f"similar_props : {similar_props}, {len(similar_props)}")

#             num_prop = random.randint(3, 10)

#             if rand_props[0] == "Nothing to Conjuct":

#                 print(f"*** No Random Property to Conjuct ***")
#                 num_rand_prop = 0
#                 num_similar_prop = num_prop

#             else:
#                 num_rand_prop = num_prop // 2
#                 num_similar_prop = num_prop - num_rand_prop

#             print(f"num_prop : {num_prop}")
#             print(f"num_rand_prop : {num_rand_prop}")
#             print(f"num_similar_prop : {num_similar_prop}")

#             if num_rand_prop != 0:

#                 if num_rand_prop <= len(rand_props):
#                     rand_props_to_conjuct = rand_props[0:num_rand_prop]
#                 else:
#                     print(
#                         f"*** Not enough Random property to conjuct, taking all Random props ***"
#                     )
#                     rand_props_to_conjuct = rand_props
#                     remaining_random_props_to_conjuct = num_rand_prop - len(rand_props)

#                     print(
#                         f"remaining_random_props_to_conjuct : {remaining_random_props_to_conjuct}"
#                     )
#                     print(f"num_similar_prop : {num_similar_prop}")

#                     num_similar_prop = (
#                         num_similar_prop + remaining_random_props_to_conjuct
#                     )

#                     print(f"num_similar_prop : {num_similar_prop}")
#             else:
#                 rand_props_to_conjuct = []

#             if num_similar_prop <= len(similar_props):
#                 similar_props_to_conjuct = similar_props[0:num_similar_prop]
#             else:
#                 print(
#                     f"*** Not enough Similar property to conjuct, taking all Similar props ***"
#                 )
#                 similar_props_to_conjuct = similar_props

#             print(f"rand_props_to_conjuct : {rand_props_to_conjuct}")
#             print(f"similar_props_to_conjuct : {similar_props_to_conjuct}")

#             all_props_to_conjuct = similar_props_to_conjuct + rand_props_to_conjuct

#             random.shuffle(all_props_to_conjuct)

#             print(f"all_props_to_conjuct : {all_props_to_conjuct}")
#             print()

#             all_random_and_similar.append(
#                 [concept, ",".join(all_props_to_conjuct), prop]
#             )

#     all_random_and_similar_df = pd.DataFrame.from_records(all_random_and_similar)

#     file_name = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/random_and_similar_conjuct_properties.tsv"

#     all_random_and_similar_df.to_csv(file_name, index=None, header=None, sep="\t")


# def prepare_joint_encoder_pretraining_data(input_file_path, strategy):

#     print(f"Input Data Path : {input_file_path}")
#     print(f"Sampling Strategy : {strategy}")

#     input_df = pd.read_csv(
#         input_file_path, header=None, sep="\t", names=["concept", "property"]
#     )

#     input_df["property"] = input_df["property"].str.replace(".", "").str.strip()

#     print(f"Shape of Input Data : {input_df.shape}")
#     print(f"Sample Inpute Data")
#     print(input_df.head(n=10))
#     print()

#     if strategy == "random":
#         random_data = random_conjuct_properties(input_df=input_df)

#     elif strategy == "similar":

#         predict_prop_embed_pkl_file = "/scratch/c.scmag3/biencoder_concept_property/trained_models/con_pro_embeddings/bb_gkb_cnet_plus_cnet_has_property_generated_property_embeddings.pkl"
#         property_vocab_embed_pkl_file = "/scratch/c.scmag3/biencoder_concept_property/trained_models/con_pro_embeddings/bert_base_gkb_cnet_trained_model_top_500k_mscg_properties_property_embeddings.pkl"

#         similar_data = get_predict_property_similar_properties(
#             predict_prop_embed_pkl_file, property_vocab_embed_pkl_file
#         )

#         return similar_data

#     elif strategy == "random_and_similar":

#         simila_prop_file = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/predict_property_similar_vocab_properties.tsv"
#         random_prop_file = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/random_conjuct_property.tsv"

#         random_and_similar_data = random_and_similar_conjuct_properties(
#             input_df=input_df,
#             similar_prop_file=simila_prop_file,
#             random_prop_file=random_prop_file,
#         )


# # input_file_path = (
# #     "train_data/gkb_source_analysis/train_gkbcnet_plus_cnethasproperty.tsv"
# # )

# input_file_path = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/gkb_source_analysis/train_gkbcnet_plus_cnethasproperty.tsv"

# # strategy = "random_and_similar"
# # prepare_joint_encoder_pretraining_data(input_file_path=input_file_path, strategy=strategy)


# strategy_list = ["random", "similar", "random_and_similar"]
# for strategy in strategy_list:

#     print(f"Sampling Data With Strategy : {strategy}")
#     prepare_joint_encoder_pretraining_data(
#         input_file_path=input_file_path, strategy=strategy
#     )
#     print(f"Finished Sampling Data With Strategy : {strategy}")
#     print(f"******************************")

