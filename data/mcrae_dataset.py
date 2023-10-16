import logging
from os import sep

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data.concept_property_dataset import TOKENIZER_CLASS


log = logging.getLogger(__name__)


class McRaeConceptPropertyDataset(Dataset):
    def __init__(self, dataset_params, dataset_type, data_df=None):
        if dataset_type in ("train", "valid"):
            self.data_df = data_df
            self.data_df.drop_duplicates(inplace=True)
            self.data_df.dropna(inplace=True)

        elif dataset_type in ("test",):
            if data_df is not None:
                log.info(f"Loading the data from supplied DF")
                self.data_df = data_df
            else:
                log.info(
                    f"*** Loading the Test Data from 'test_file_path', DF supplied is None ***"
                )
                self.data_df = pd.read_csv(
                    dataset_params.get("test_file_path"),
                    sep="\t",
                    header=None,
                    names=["concept", "property", "label"],
                )

                self.data_df.drop_duplicates(inplace=True)
                self.data_df.dropna(inplace=True)
                self.data_df.reset_index(drop=True, inplace=True)

            log.info(f"Test Data size {self.data_df.shape}")

        self.hf_tokenizer_name = dataset_params.get("hf_tokenizer_name")

        self.tokenizer_class = TOKENIZER_CLASS.get(self.hf_tokenizer_name)

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = self.tokenizer_class.from_pretrained(
            dataset_params.get("hf_tokenizer_path")
        )

        self.mask_token = self.tokenizer.mask_token
        self.concept_max_len = dataset_params.get("concept_max_len", 510)
        self.property_max_len = dataset_params.get("property_max_len", 510)

        self.context_num = dataset_params.get("context_num")

        self.label = self.data_df["label"].values

        log.info(f"hf_tokenizer_name : {dataset_params.get('hf_tokenizer_name')}")
        log.info(f"self.tokenizer_class : {self.tokenizer_class}")
        log.info(f"Mask Token for the Model : {self.mask_token}")
        log.info(f"Context Num : {self.context_num}")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return [
            self.data_df["concept"][idx],
            self.data_df["property"][idx],
            self.data_df["label"][idx],
        ]

    def add_context(self, batch):
        ############### The Following Input Template uses Mean Strategy ###############
        if self.context_num == 1:
            concept_context = "Concept : "
            property_context = "Property : "

            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]

        elif self.context_num == 2:
            concept_context = "The notion we are modelling is "
            property_context = "The notion we are modelling is "

            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]

        elif self.context_num == 3:
            prefix_num = 5
            suffix_num = 4

            print("prefix_num :", prefix_num)
            print("suffix_num :", suffix_num)

            concepts_batch = [
                "[MASK] " * prefix_num + concept + " " + "[MASK] " * suffix_num + "."
                for concept in batch[0]
            ]
            property_batch = [
                "[MASK] " * prefix_num + prop + " " + "[MASK] " * suffix_num + "."
                for prop in batch[1]
            ]
        elif self.context_num == 4:
            concept_context = "Yesterday, I saw another "
            property_context = "Yesterday, I saw a thing which is "

            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]

        elif self.context_num == 5:
            concept_context = "The notion we are modelling is called "
            property_context = "The notion we are modelling is called "

            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]

        ############### The Following Input Template uses Mask Strategy ###############

        elif self.context_num == 6:
            # [CLS] CONCEPT means [MASK] [SEP]
            # context = " means [MASK]"

            # concepts_batch = [x + context for x in batch[0]]
            # property_batch = [x + context for x in batch[1]]

            context = " means " + self.mask_token

            # concepts_batch = [x.strip().replace(".", "") + context for x in batch[0]]
            # property_batch = [x.strip().replace(".", "") + context for x in batch[1]]

            concepts_batch = [x.strip() + context for x in batch[0]]
            property_batch = [x.strip() + context for x in batch[1]]

        elif self.context_num == 7:
            # [CLS] CONCEPT [SEP] [MASK] [SEP]

            concepts_batch = [x for x in batch[0]]
            property_batch = [x for x in batch[1]]

        elif self.context_num == 8:
            # [CLS] The notion we are modelling is CONCEPT. [SEP] [MASK] [SEP]

            context = "The notion we are modelling is "

            concepts_batch = [context + x + "." for x in batch[0]]
            property_batch = [context + x + "." for x in batch[1]]

        elif self.context_num == 9:
            # [CLS] The spaceship we are modelling is CONCEPT. [SEP] [MASK] [SEP]

            context = "The spaceship we are modelling is "

            concepts_batch = [context + x + "." for x in batch[0]]
            property_batch = [context + x + "." for x in batch[1]]

        elif self.context_num == 10:
            # [CLS] We are modelling CONCEPT.[SEP] [MASK] [SEP]

            context = "We are modelling "

            concepts_batch = [context + x + "." for x in batch[0]]
            property_batch = [context + x + "." for x in batch[1]]

        elif self.context_num == 11:
            # [CLS] The notion we are modelling this morning is CONCEPT.[SEP][MASK][SEP]

            context = "The notion we are modelling this morning is "

            concepts_batch = [context + x + "." for x in batch[0]]
            property_batch = [context + x + "." for x in batch[1]]

        elif self.context_num == 12:
            # [CLS] As I have mentioned earlier, the notion we are modelling this morning is CONCEPT.[SEP][MASK][SEP]

            context = "As I have mentioned earlier, the notion we are modelling this morning is "

            concepts_batch = [context + x + "." for x in batch[0]]
            property_batch = [context + x + "." for x in batch[1]]

        return concepts_batch, property_batch

    def tokenize(
        self, concept_batch, property_batch, concept_max_len=64, property_max_len=64
    ):
        # if self.context_num in (1, 2, 3, 4, 5, 6):

        # # Printing for debugging
        # print(f"Context Num : {self.context_num}")
        # print("concept_batch :", concept_batch)
        # print("property_batch :", property_batch)
        # print()

        concept_ids = self.tokenizer(
            concept_batch,
            add_special_tokens=True,
            max_length=self.concept_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        property_ids = self.tokenizer(
            property_batch,
            add_special_tokens=True,
            max_length=self.property_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # # Printing for debugging
        # print("concept_ids")
        # print(concept_ids.get("input_ids"))
        # print("concept_token_type_id")
        # print(concept_ids.get("token_type_ids"))

        # for i in concept_ids.get("input_ids"):
        #     print(self.tokenizer.convert_ids_to_tokens(torch.tensor(i)))

        # print()
        # print("property_inp_id")
        # print(property_ids.get("input_ids"))
        # print("property_token_type_id")
        # print(property_ids.get("token_type_ids"))

        # for i in property_ids.get("input_ids"):
        #     print(self.tokenizer.convert_ids_to_tokens(torch.tensor(i)))
        # print("*" * 50)

        # else:

        #     context_second_sent = ["[MASK]" for i in range(len(concept_batch))]
        #     property_second_sent = ["[MASK]" for i in range(len(concept_batch))]

        #     # # Printing for debugging
        #     # print("*" * 50, flush=True)
        #     # print(f"Context Num : {self.context_num}")
        #     # print("concept_batch :", concept_batch)
        #     # print("context_second_sent :", context_second_sent)
        #     # print("property_batch :", property_batch)
        #     # print("property_second_sent :", property_second_sent)

        #     concept_ids = self.tokenizer(
        #         concept_batch,
        #         context_second_sent,
        #         add_special_tokens=True,
        #         max_length=concept_max_len,
        #         padding=True,
        #         truncation=True,
        #         return_tensors="pt",
        #     )

        #     property_ids = self.tokenizer(
        #         property_batch,
        #         property_second_sent,
        #         add_special_tokens=True,
        #         max_length=property_max_len,
        #         padding=True,
        #         truncation=True,
        #         return_tensors="pt",
        #     )

        # # Printing for debugging
        # print("concept_ids")
        # print(concept_ids.get("input_ids"))
        # print("concept_token_type_id")
        # print(concept_ids.get("token_type_ids"))

        # for i in concept_ids.get("input_ids"):
        #     print(self.tokenizer.convert_ids_to_tokens(torch.tensor(i)))

        # print()
        # print("property_inp_id")
        # print(property_ids.get("input_ids"))
        # print("property_token_type_id")
        # print(property_ids.get("token_type_ids"))

        # for i in property_ids.get("input_ids"):
        #     print(self.tokenizer.convert_ids_to_tokens(torch.tensor(i)))
        # print("*" * 50, flush=True)

        if self.hf_tokenizer_name in ("roberta-base", "roberta-large"):
            return {
                "concept_inp_id": concept_ids.get("input_ids"),
                "concept_atten_mask": concept_ids.get("attention_mask"),
                "property_inp_id": property_ids.get("input_ids"),
                "property_atten_mask": property_ids.get("attention_mask"),
            }
        else:
            return {
                "concept_inp_id": concept_ids.get("input_ids"),
                "concept_atten_mask": concept_ids.get("attention_mask"),
                "concept_token_type_id": concept_ids.get("token_type_ids"),
                "property_inp_id": property_ids.get("input_ids"),
                "property_atten_mask": property_ids.get("attention_mask"),
                "property_token_type_id": property_ids.get("token_type_ids"),
            }
