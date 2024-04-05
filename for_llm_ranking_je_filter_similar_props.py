import logging
import os
import gc
import time
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import pandas as pd
import torch

from model.lm_con_prop import prepare_data_and_models
from utils.je_utils import read_config, set_seed


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

    # test_file = training_params["test_file_path"]
    thresholds = list(training_params["thresholds"])
    save_dir = training_params["save_dir"]
    pretrained_model_path = training_params["pretrained_model_path"]
    pretrained_model_num_neg = training_params["pretrained_model_num_neg"]
    dataset_name = training_params["dataset_name"]
    pretrained_model_to_use = training_params["pretrained_model_to_use"]
    prop_applies_to_concepts = training_params["prop_applies_to_concepts"]
    create_complete_clusters = training_params["create_complete_clusters"]

    test_dir = training_params["test_file_path"]
    input_files = os.listdir(test_dir)

    log.info(f"Test Data Dir : {test_dir}")
    log.info(f"input_files: {input_files}")

    log.info(f"Filtering Threshold : {type(thresholds)}, {thresholds}")
    log.info(f"Save Dir : {save_dir}")
    log.info(f"pretrained_model_num_neg : {pretrained_model_num_neg}")
    log.info(f"Pretrained Model Path : {pretrained_model_path}")
    log.info(f"create_complete_clusters : {create_complete_clusters}")

    # if pretrained_model_to_use == "je_con_prop":

    for i, inp_file in enumerate(input_files):

        log.info(f"\nProcessing file {inp_file}, {i+1} / {len(input_files)}")

        file_path = os.path.join(test_dir, inp_file)

        log.info(f"Getting logits for file: {inp_file}")
        log.info(f"File location: {file_path}")

        test_df = pd.read_csv(file_path, sep="\t", header=None)

        num_columns = len(test_df.columns)
        log.info(f"Number of columns in input file : {num_columns}")

        if num_columns == 2:
            log.info(f"input_df")
            log.info(test_df)

            test_df["label"] = int(0)
            test_df.rename(columns={0: "concept", 1: "property"}, inplace=True)
            test_df = test_df[["concept", "property", "label"]]
        else:
            raise Exception(f"check_input_file: {file_path}")

        model, test_dataloader = prepare_data_and_models(
            config=config, train_file=None, valid_file=None, test_file=test_df
        )
        loss, predictions, logit = predict(model=model, dataloader=test_dataloader)

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
            save_dir, f"{pretrained_model_num_neg}_{inp_file}"
        )

        new_test_dataframe.to_csv(
            all_logit_filename, sep="\t", index=None, header=None, float_format="%.5f"
        )

        log.info(f"all_data - Dataframe With Logits")
        log.info(f"file_saved: {all_logit_filename}")
        log.info(new_test_dataframe.head(n=20))
        log.info("*" * 50)

        del model
        del test_df
        del new_test_dataframe
        del test_dataloader

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
