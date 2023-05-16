import logging
import os
import time
from argparse import ArgumentParser
from pprint import pprint
from math import ceil

import numpy as np
import pandas as pd
import torch

from model.lm_con_prop import prepare_data_and_models
from utils.je_utils import read_config, set_seed

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from model.lm_con_prop import (
    ModelConceptPropertyJoint,
    ModelAnyNumberLabel,
    ModelSeqClassificationConPropJoint,
)


def set_logger(config):
    log_file_name = os.path.join(
        "logs",
        config.get("log_dirctory"),
        f"log_{config.get('experiment_name')}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt",
    )

    print("config.get('experiment_name') :", config.get("experiment_name"))
    print("\nlog_file_name :", log_file_name)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file_name,
        filemode="w",
        format="%(asctime)s : %(name)s : %(levelname)s - %(message)s",
    )


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# test_file = "data/generate_embeddding_data/mcrae_related_data/dummy.txt"
# test_file = "data/train_data/joint_encoder_property_conjuction_data/false_label_con_similar_50_vocab_props.txt"

# test_file = (
#     "data/redo_prop_conj_exp/with_false_labels_cnetp_con_similar_50_prop_vocab.tsv"
# )


# model_name = (
#     "joint_encoder_concept_property_gkbcnet_cnethasprop_step2_pretrained_model.pt"
# )
# model_name = "je_con_prop_cnet_premium_10negdata_pretrained_model.pt"
# model_name = "je_con_prop_cnet_premium_20negdata_pretrained_model.pt"


def predict(model, dataloader):
    model.eval()
    model.to(device)

    test_loss, test_preds, test_logits = [], [], []

    for step, batch in enumerate(dataloader):
        log.info(f"Processing batch: {step} of {len(dataloader)}")

        input_ids = batch["input_ids"].squeeze().to(device)
        token_type_ids = batch["token_type_ids"].squeeze().to(device)
        attention_mask = batch["attention_mask"].squeeze().to(device)

        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        if isinstance(model, ModelConceptPropertyJoint) or isinstance(
            model, ModelAnyNumberLabel
        ):
            loss, logits, mask_vectors = outputs
            batch_preds = torch.round(torch.sigmoid(logits))

            # print(f"logits : {logits}")
            # print(
            #     f"sigmoid(logits) : {torch.sigmoid(logits).shape}, {torch.sigmoid(logits)}"
            # )

            test_logits.extend(torch.sigmoid(logits).cpu().numpy())

        elif isinstance(model, ModelSeqClassificationConPropJoint):
            loss, logits = outputs

            batch_probs = logits.softmax(dim=1).squeeze(0)
            batch_preds = torch.argmax(batch_probs, dim=1).flatten()

            ######### - Check this
            positive_class_logits = [l[1] for l in logit]
            test_logits.extend(torch.sigmoid(positive_class_logits).cpu().numpy())
            #########

        test_loss.append(loss.cpu().numpy())
        test_preds.extend(batch_preds.cpu().numpy())

    loss = np.mean(test_loss)

    return loss, test_preds, test_logits


if __name__ == "__main__":
    set_seed(1)

    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", required=True, help="configuration file path"
    )
    args = parser.parse_args()

    config = read_config(args.config_file)

    set_logger(config=config)
    log = logging.getLogger(__name__)

    log.info("The model is run with the following configuration")
    log.info(f"\n {config} \n")
    pprint(config)

    training_params = config["training_params"]

    test_file = training_params["test_file_path"]
    threshold = training_params["threshold"]
    save_dir = training_params["save_dir"]
    pretrained_model_path = training_params["pretrained_model_path"]
    pretrained_model_num_neg = training_params["pretrained_model_num_neg"]
    dataset_name = training_params["dataset_name"]
    pretrained_model_to_use = training_params["pretrained_model_to_use"]

    log.info(f"Test File : {test_file}")
    log.info(f"Filtering Threshold : {threshold}")
    log.info(f"Save Dir : {save_dir}")
    log.info(f"pretrained_model_num_neg : {pretrained_model_num_neg}")
    log.info(f"Pretrained Model Path : {pretrained_model_path}")

    if pretrained_model_to_use == "je_con_prop":
        test_df = pd.read_csv(
            test_file,
            sep="\t",
            header=None,
            names=["concept", "property", "label"],
        )

        log.info(f"Test Df")
        log.info(test_df)

        model, test_dataloader = prepare_data_and_models(
            config=config, train_file=None, valid_file=None, test_file=test_file
        )

        loss, predictions, logit = predict(model=model, dataloader=test_dataloader)

        # positive_class_logits = [l[1] for l in logit]

        log.info(f"Number of Logits : {len(logit)}")

        assert test_df.shape[0] == len(
            logit
        ), f"length of test dataframe, {test_df.shape[0]} is not equal to logits, {len(logit)}"

        new_test_dataframe = test_df.copy(deep=True)
        new_test_dataframe.drop("label", axis=1, inplace=True)
        new_test_dataframe["logit"] = logit

        log.info(f"new_test_dataframe")
        log.info(new_test_dataframe.head(n=20))

        all_logit_filename = os.path.join(
            save_dir,
            f"{pretrained_model_num_neg}_with_logits_all_data_{dataset_name}.tsv",
        )
        new_test_dataframe.to_csv(
            all_logit_filename, sep="\t", index=None, header=None, float_format="%.5f"
        )

        log.info(f"All Data - Dataframe With Logits")
        log.info(new_test_dataframe.head(n=20))

        df_with_threshold = new_test_dataframe[new_test_dataframe["logit"] > threshold]

        with_threshold_logit_filename = os.path.join(
            save_dir,
            f"{pretrained_model_num_neg}_{threshold}thres_filtered_with_logits_{dataset_name}.tsv",
        )

        df_with_threshold.to_csv(
            with_threshold_logit_filename,
            sep="\t",
            index=None,
            header=None,
            float_format="%.5f",
        )

        log.info(f"Threshold {threshold} Data - Dataframe With Logits")
        log.info(df_with_threshold.head(n=20))

        df_with_threshold.drop(labels="logit", axis=1, inplace=True)

        logit_filename = os.path.join(
            save_dir,
            f"{pretrained_model_num_neg}_{threshold}thres_filtered_without_logits_conprop_{dataset_name}.tsv",
        )
        df_with_threshold.to_csv(logit_filename, sep="\t", index=None, header=None)

        log.info(f"Data after logit column is dropped")
        log.info(df_with_threshold.head(n=20))

    elif pretrained_model_to_use == "nli":
        nli_tokenizer_path = training_params["nli_tokenizer_path"]
        nli_model_path = training_params["nli_model_path"]
        batch_size = training_params["batch_size"]

        tokenizer = AutoTokenizer.from_pretrained(nli_tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(nli_model_path).to(
            device
        )

        test_df = pd.read_csv(
            test_file,
            sep="\t",
            header=None,
            names=["concept", "property"],
        )

        log.info(f"Test Df")
        log.info(test_df)

        all_data_filename = os.path.join(
            save_dir,
            f"{pretrained_model_num_neg}_nli_{dataset_name}_with_all_classess.tsv",
        )

        filtered_data_filename = os.path.join(
            save_dir,
            f"{pretrained_model_num_neg}_nli_{dataset_name}_with_entailed_class.tsv",
        )

        log.info(f"all_data_filename : {all_data_filename}")
        log.info(f"filtered_data_filename : {filtered_data_filename}")

        label_names = ["entailment", "neutral", "contradiction"]
        id2label = {id: label for id, label in enumerate(label_names)}

        batch_counter = 0

        with open(all_data_filename, "w") as all_file, open(
            filtered_data_filename, "w"
        ) as entailed_file:
            for idx in range(0, len(test_df), batch_size):
                log.info(
                    f"Processing Batch : {batch_counter} / {ceil(len(test_df) / batch_size)}"
                )

                end_idx = idx + batch_size

                concept = test_df["concept"].to_list()[idx:end_idx]
                property = test_df["property"].to_list()[idx:end_idx]

                input = tokenizer(
                    concept,
                    property,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    output = model(input["input_ids"].to(device))

                probs = torch.softmax(output["logits"], -1)
                conf = [torch.round(pred * 100) for pred in probs]

                label = torch.argmax(probs, dim=1)
                predict_class = [id2label[l.item()] for l in label]

                line_data = [
                    (p, h, str(conf.cpu().tolist()), cl)
                    for p, h, conf, cl in zip(concept, property, conf, predict_class)
                ]

                for item in line_data:
                    line_to_write = "\t".join(item)
                    all_file.write("{0}\n".format(line_to_write))
                    print(line_to_write, flush=True)

                    if item[3] == "entailment":
                        con_prop_entailed = item[0:2]
                        con_prop_entailed = "\t".join(con_prop_entailed)

                        entailed_file.write("{0}\n".format(con_prop_entailed))

                batch_counter += 1
                log.info(f"Records Processed:  {end_idx} / {len(test_df)}")

        log.info(f"All data with NLI classes saved in : {all_data_filename}")
        log.info(f"Entailed data saved in : {filtered_data_filename}")
