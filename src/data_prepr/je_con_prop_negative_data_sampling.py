import os
import sys
import pandas as pd
from pprint import pprint

sys.path.insert(0, os.getcwd())


def negative_sampling(df, data_type, num_negative=5):

    pos_data_list = df.values.tolist()

    print()
    print("*" * 50, flush=True)
    print(f"Generating - {data_type} Positive Negative Data", flush=True)
    print(df.shape, flush=True)
    pprint(df.head())
    print()

    unique_concepts = df["concept"].unique()
    unique_properties = df["property"].unique()

    df["concept"] = df["concept"].str.replace(".", "").str.strip()
    df["property"] = df["property"].str.replace(".", "").str.strip()

    total_unique_concepts = len(unique_concepts)
    print(
        f"Number of Unique Concepts in Dataframe :", total_unique_concepts, flush=True
    )
    print(
        f"Number of Unique Property in Dataframe :", len(unique_properties), flush=True
    )

    all_negative_data = []

    for i, concept in enumerate(unique_concepts, start=1):

        print(f"Processing Concept: {concept} -  {i} of {total_unique_concepts}")

        concept_data = df[df["concept"] == concept]
        properties_for_concept = concept_data["property"].unique()

        if concept == "humans":
            # Concept - humans was going into infinite while loop so skiping genetaing negative for it, for now.
            print(f"Skiping concept : {concept}")
            continue

        num_record = len(concept_data)

        print()
        print(f"Generating Negative Data for Concept : {concept}", flush=True)
        print(f"Positive data for concept in DF : {concept_data.shape}", flush=True)

        print("Data For concept", flush=True)
        pprint(concept_data)
        print(f"Properties for Concept", flush=True)
        print(properties_for_concept, flush=True)

        total_neg_num = num_record * num_negative

        print(f"Total Number of Negative Records to be generated : {total_neg_num}")

        rest_df = df[~(df["concept"] == concept)]
        print(f"Rest DF shape after removing concept : {rest_df.shape}")

        rest_df = rest_df[~rest_df["property"].isin(properties_for_concept)]
        print(
            f"Rest DF shape after removing concepts's properties : {rest_df.shape}",
            flush=True,
        )

        concept_neg_data = []

        while True:

            concept = concept.strip()
            neg_properties = list(rest_df["property"].sample(n=total_neg_num))

            neg_data = [[concept, neg_prop] for neg_prop in neg_properties]
            print(f"neg_data length :", len(neg_data), flush=True)

            if len(concept_neg_data) < total_neg_num:
                for x in neg_data:
                    if not (x in pos_data_list):
                        if not (x in all_negative_data):

                            all_negative_data.append(x)
                            concept_neg_data.append(x)

                            if len(concept_neg_data) == total_neg_num:
                                break

            if len(concept_neg_data) == total_neg_num:
                break

        print(
            f"Number of negative records generated : {len(concept_neg_data)}",
            flush=True,
        )
        print(f"Negative Records", flush=True)
        print(concept_neg_data, flush=True)
        print()

    _ = [x.insert(2, int(0)) for x in all_negative_data]

    # print ("all_negative_data")
    # print (all_negative_data)

    all_neg_data_df = pd.DataFrame.from_records(
        all_negative_data, columns=["concept", "property", "label"]
    )

    neg_data_duplicate_records = all_neg_data_df[
        all_neg_data_df.duplicated(["concept", "property"])
    ]

    print()
    print(f"all_neg_data_df.shape : {all_neg_data_df.shape}", flush=True)
    print(
        f"neg_data_duplicate_records.shape : {neg_data_duplicate_records.shape}",
        flush=True,
    )
    print()

    print(f"Checking overlap between positive and negative data", flush=True)
    pos_neg_overlap_df = df.merge(
        all_neg_data_df, how="inner", on=["concept", "property"], indicator=False
    )
    print(f"Positive and Negative Overlapped Dataframe", flush=True)
    print(pos_neg_overlap_df, flush=True)
    print()

    df["label"] = int(1)
    pos_neg_df = pd.concat([df, all_neg_data_df], axis=0, ignore_index=True)

    print("DF after adding negative data", flush=True)
    print(pos_neg_df.shape, flush=True)

    duplicate_records = pos_neg_df[pos_neg_df.duplicated(["concept", "property"])]

    print(f"Duplicate Records : {duplicate_records.shape}", flush=True)
    print(
        f"Duplicate record label value count: {duplicate_records['label'].value_counts()}",
        flush=True,
    )
    print()

    pos_neg_df = pos_neg_df[
        ~pos_neg_df.duplicated(subset=["concept", "property"], keep="first")
    ]

    pos_neg_df.drop_duplicates(inplace=True)
    pos_neg_df.dropna(how="any", inplace=True)

    pos_neg_df.dropna(axis=0, subset=["concept"], inplace=True)
    pos_neg_df.dropna(axis=0, subset=["property"], inplace=True)
    pos_neg_df.dropna(axis=0, subset=["label"], inplace=True)

    pos_neg_df = pos_neg_df.sample(frac=1)

    print(
        f"{data_type} - {num_neg} Neg Dataframe after removing duplicates : {pos_neg_df.shape}",
        flush=True,
    )

    pprint(pos_neg_df)

    return pos_neg_df


# local_train_file = "data/train_data/joint_encoder_concept_property_data/original_train_gkbcnet_plus_cnethasproperty.tsv"
# local_val_file = "data/train_data/joint_encoder_concept_property_data/original_valid_gkbcnet_plus_cnethasproperty.tsv"

# hawk_train_file = "data/train_data/gkb_source_analysis/train_pos_cnetp.tsv"
# hawk_val_file = "data/train_data/gkb_source_analysis/valid_pos_cnetp.tsv"

hawk_train_file = "/scratch/c.scmag3/hawk_data/biencoder_concept_property/data/train_data/gkb_source_analysis/train_gkbcnet_plus_cnethasproperty_plus_mscg.tsv"
hawk_val_file = "/scratch/c.scmag3/hawk_data/biencoder_concept_property/data/train_data/gkb_source_analysis/valid_gkbcnet_plus_cnethasproperty_plus_mscg.tsv"


train_df = pd.read_csv(
    hawk_train_file, header=None, names=["concept", "property"], sep="\t"
)
valid_df = pd.read_csv(
    hawk_val_file, header=None, names=["concept", "property"], sep="\t"
)


num_neg_pair = [5]
for num_neg in num_neg_pair:

    print()
    print("*" * 50)
    print(f"Generating Negative Train Data for num negative : {num_neg}")

    print("Training Data")

    pos_neg_train_df = negative_sampling(
        train_df, data_type="train", num_negative=num_neg
    )

    base_path = "data/train_data/je_con_prop"
    save_train_file_name = os.path.join(
        base_path, f"{num_neg}_neg_train_mscg_cnetp.tsv"
    )
    save_valid_file_name = os.path.join(
        base_path, f"{num_neg}_neg_valid_mscg_cnetp.tsv"
    )

    pos_neg_train_df.to_csv(save_train_file_name, sep="\t", index=None, header=None)

    print("New Train 0/1 Labels")
    print(pos_neg_train_df["label"].value_counts())

    print("Train Record Before Negative Data:", len(train_df))
    print("Train Record After Negative Data:", len(pos_neg_train_df))

    print(f"Generating Negative Valid Data for num negative : {num_neg}")
    print("Validation Data")

    pos_neg_val_df = negative_sampling(
        valid_df, data_type="valid", num_negative=num_neg
    )
    pos_neg_val_df.to_csv(save_valid_file_name, sep="\t", index=None, header=None)

    print("New Validation 0/1 Labels")
    print(pos_neg_val_df["label"].value_counts())

    print()
    print("Validation Record Before Negative Data:", len(valid_df))
    print("Validation Record After Negative Data:", len(pos_neg_val_df))

    print(
        f"Finished generating negative data for Train/Validation data for num negative : {num_neg}"
    )

    print("*" * 50)
    print()
