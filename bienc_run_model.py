import argparse
import logging
import os
import math


import numpy as np
import torch
import torch.nn as nn
from tqdm.std import trange
from pprint import pprint
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.functions import (
    compute_scores,
    create_dataset_and_dataloader,
    create_model,
    read_config,
    set_seed,
    calculate_loss,
)

log = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(
    model, train_dataset, train_dataloader, loss_fn, optimizer, scheduler
):
    epoch_loss = 0.0

    model.train()

    print_freq = 0

    for step, batch in enumerate(train_dataloader):
        model.zero_grad()

        concepts_batch, property_batch = train_dataset.add_context(batch)

        if print_freq < 1:
            log.info(f"concepts_batch : {concepts_batch}")
            log.info(f"property_batch : {property_batch}")

            print(f"concepts_batch : {concepts_batch}", flush=True)
            print(f"property_batch : {property_batch}", flush=True)

            print_freq += 1

        ids_dict = train_dataset.tokenize(concepts_batch, property_batch)

        # log.info(f"\n")
        # log.info(f"******************************")

        # for key, value in ids_dict.items():
        #     log.info(f"{key} : {value}")

        # log.info(f"******************************")
        # log.info(f"\n")

        if train_dataset.hf_tokenizer_name in ("roberta-base", "roberta-large"):
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

        concept_embedding, property_embedding, logits = model(
            concept_input_id=concept_inp_id,
            concept_attention_mask=concept_attention_mask,
            concept_token_type_id=concept_token_type_id,
            property_input_id=property_input_id,
            property_attention_mask=property_attention_mask,
            property_token_type_id=property_token_type_id,
        )

        batch_loss, batch_logits, batch_labels = calculate_loss(
            dataset=train_dataset,
            batch=batch,
            concept_embedding=concept_embedding,
            property_embedding=property_embedding,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_loss += batch_loss.item()

        batch_loss.backward()
        torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if step % 100 == 0 and not step == 0:
            batch_labels = batch_labels.reshape(-1, 1).detach().cpu().numpy()

            batch_logits = (
                torch.round(torch.sigmoid(batch_logits))
                .reshape(-1, 1)
                .detach()
                .cpu()
                .numpy()
            )

            batch_scores = compute_scores(batch_labels, batch_logits)

            log.info(
                f"Batch {step} of {len(train_dataloader)} ----> Batch Loss : {batch_loss}, Batch Binary F1 {batch_scores.get('binary_f1')}"
            )
            print(
                f"Batch {step} of {len(train_dataloader)} ----> Batch Loss : {batch_loss}, Batch Binary F1 {batch_scores.get('binary_f1')}",
                flush=True,
            )
            print(flush=True)

    avg_epoch_loss = epoch_loss / len(train_dataloader)

    return avg_epoch_loss


def evaluate(model, valid_dataset, valid_dataloader, loss_fn, device):
    val_loss = 0.0

    model.eval()

    epoch_logits, epoch_labels = [], []

    for step, batch in enumerate(valid_dataloader):
        concepts_batch, property_batch = valid_dataset.add_context(batch)

        ids_dict = valid_dataset.tokenize(concepts_batch, property_batch)

        if valid_dataset.hf_tokenizer_name in ("roberta-base", "roberta-large"):
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

        batch_loss, batch_logits, batch_labels = calculate_loss(
            dataset=valid_dataset,
            batch=batch,
            concept_embedding=concept_embedding,
            property_embedding=property_embedding,
            loss_fn=loss_fn,
            device=device,
        )  # dataset, batch, concept_embedding, property_embedding, loss_fn, device

        epoch_logits.append(batch_logits)
        epoch_labels.append(batch_labels)

        val_loss += batch_loss.item()
        torch.cuda.empty_cache()

    epoch_logits = (
        torch.round(torch.sigmoid(torch.vstack(epoch_logits)))
        .reshape(-1, 1)
        .detach()
        .cpu()
        .numpy()
    )

    epoch_labels = torch.vstack(epoch_labels).reshape(-1, 1).detach().cpu().numpy()

    scores = compute_scores(epoch_labels, epoch_logits)

    avg_val_loss = val_loss / len(valid_dataloader)

    return avg_val_loss, scores


def train(config):
    log.info("Initialising datasets...")

    train_dataset, train_dataloader = create_dataset_and_dataloader(
        config.get("dataset_params"), dataset_type="train"
    )

    valid_dataset, valid_dataloader = create_dataset_and_dataloader(
        config.get("dataset_params"), dataset_type="valid"
    )

    log.info("Initialising Model...")

    model = create_model(config.get("model_params"))
    model.to(device)
    # log.info(f"Model Loaded : {model}")

    # -------------------- Preparation for training  ------------------- #

    weight_decay = config["training_params"]["weight_decay"]

    loss_fn = nn.BCEWithLogitsLoss()

    lr = config["training_params"]["lr"]
    weight_decay = config["training_params"]["weight_decay"]

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_training_steps = len(train_dataloader) * config["training_params"].get(
        "max_epochs"
    )

    if config["training_params"]["lr_policy"] == "warmup":
        warmup_ratio = config["training_params"]["warmup_ratio"]
        num_warmup_steps = math.ceil(total_training_steps * warmup_ratio)

    else:
        num_warmup_steps = 0

    log.info(f"Warmup-steps: {num_warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_f1 = 0.0
    start_epoch = 1

    epoch_count = []
    train_losses = []
    valid_losses = []

    log.info(f"Training the concept property model on {device}")

    patience_counter = 0

    for epoch in trange(start_epoch, config["training_params"].get("max_epochs") + 1):
        log.info(f"  Epoch {epoch} of {config['training_params'].get('max_epochs')}")
        print("\n", flush=True)

        train_loss = train_single_epoch(
            model=model,
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        log.info(f"Train Epoch {epoch} finished !!")
        log.info(f"  Average Train Loss: {train_loss}")

        # ----------------------------------------------#
        # ----------------------------------------------#
        # ---------------Validation---------------------#
        # ----------------------------------------------#
        # ----------------------------------------------#

        log.info(f"Running Validation ....")
        print(flush=True)

        valid_loss, valid_scores = evaluate(
            model=model,
            valid_dataset=valid_dataset,
            valid_dataloader=valid_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_count.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        log.info(f"  Average validation Loss: {valid_loss}")
        print(flush=True)

        val_binary_f1 = valid_scores.get("binary_f1")

        if val_binary_f1 < best_val_f1:
            patience_counter += 1
        else:
            patience_counter = 0
            best_val_f1 = val_binary_f1

            best_model_path = os.path.join(
                config["training_params"].get("export_path"),
                config["model_params"].get("model_name"),
            )

            log.info(f"patience_counter : {patience_counter}")
            log.info(f"best_model_path : {best_model_path}")

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            log.info(f"Best model at epoch: {epoch}, Binary F1: {val_binary_f1}")
            log.info(f"The model is saved in : {best_model_path}")

        log.info("Validation Scores")
        log.info(f" Best Validation F1 yet : {best_val_f1}")

        for key, value in valid_scores.items():
            log.info(f"{key} : {value}")

        print(flush=True)

        print("train_losses", flush=True)
        print(train_losses, flush=True)
        print("valid_losses", flush=True)
        print(valid_losses, flush=True)

        if patience_counter >= config["training_params"].get("early_stopping_patience"):
            log.info(
                f"Early Stopping ---> Maximum Patience - {config['training_params'].get('early_stopping_patience')} Reached !!"
            )
            break

        print(flush=True)


def test_best_model(config):
    log.info(f"\n {'*' * 50}")
    log.info(f"Testing the best model")

    model = create_model(config.get("model_params"))

    best_model_path = os.path.join(
        config["training_params"]["export_path"],
        config["model_params"]["model_name"],
    )

    log.info(f"Testing the best model : {best_model_path}")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)

    test_dataset, test_dataloader = create_dataset_and_dataloader(
        config.get("dataset_params"), dataset_type="test"
    )

    label = test_dataset.label
    all_test_preds = []

    for step, batch in enumerate(test_dataloader):
        concepts_batch, property_batch = test_dataset.add_context(batch)

        ids_dict = test_dataset.tokenize(concepts_batch, property_batch)

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

        preds = torch.round(torch.sigmoid(logits))
        all_test_preds.extend(preds.detach().cpu().numpy().flatten())

    test_scores = compute_scores(label, np.asarray(all_test_preds))

    log.info(f"Test Metrices")
    log.info(f"Test labels shape: {label.shape}")
    log.info(f"Test Preds shape: {np.asarray(all_test_preds).shape}")

    for key, value in test_scores.items():
        log.info(f"{key} : {value}")
    print(flush=True)


if __name__ == "__main__":
    # set_seed(12345)
    set_seed(1)

    parser = argparse.ArgumentParser(description="Biencoder Concept Property Model")

    parser.add_argument(
        "--config_file",
        required=True,
        help="path to the configuration file",
    )

    args = parser.parse_args()

    log.info(f"Reading Configuration File: {args.config_file}")
    config = read_config(args.config_file)

    log.info("The model is run with the following configuration")

    log.info(f"\n {config} \n")
    print(f"Input Config File")
    pprint(config, sort_dicts=False)

    hp_tuning = config["training_params"]["hp_tuning"]

    if not hp_tuning:
        train(config)

        # We are not testing the model yet..We will test it on McRae Testset after finetuning
        # test_best_model(config)

    else:
        # bienc_chatgpt20k_pretrain_bert_base_uncased_ep14_bs8_wr0_wd0.1_lr2e-06_do0.1.pt
        log.info("Doing Hyperparameter Search With Grid Search")

        max_epochs = [18]
        batch_size = [8]
        warmup_ratio = [0, 0.6, 0.1, 0.15]
        weight_decay = [0.1, 0.01, 0.2]

        lr = [2e-6]
        hidden_dropout_prob = [0.1, 0.3]

        log.info(f"max_epochs : {max_epochs}")
        log.info(f"batch_size : {batch_size}")
        log.info(f"warmup_ratio : {warmup_ratio}")
        log.info(f"weight_decay : {weight_decay}")

        log.info(f"lr : {lr}")
        log.info(f"hidden_dropout_prob : {hidden_dropout_prob}")

        hf_checkpoint_name = config["model_params"]["hf_checkpoint_name"]
        model_prefix = config["model_params"]["model_name"]

        for me in max_epochs:
            for bs in batch_size:
                for wr in warmup_ratio:
                    for wd in weight_decay:
                        for l in lr:
                            for do in hidden_dropout_prob:
                                discription_str = (
                                    f"ep{me}_bs{bs}_wr{wr}_wd{wd}_lr{l}_do{do}"
                                )

                                config["training_params"]["max_epochs"] = me
                                config["dataset_params"]["loader_params"][
                                    "batch_size"
                                ] = bs
                                config["training_params"]["warmup_ratio"] = wr
                                config["training_params"]["weight_decay"] = wd

                                config["training_params"]["lr"] = l
                                config["model_params"]["hidden_dropout_prob"] = do

                                config["model_params"]["model_name"] = (
                                    model_prefix
                                    + "_"
                                    + hf_checkpoint_name.replace("-", "_")
                                    + "_"
                                    + discription_str
                                    + ".pt"
                                )

                                log.info("\n")
                                log.info("*" * 50)

                                log.info(f"discription_str : {discription_str}")

                                log.info(
                                    f"new_model_run : max_epochs: {me}, batch_size: {bs}, warmup_ratio : {wr}, weight_decay : {wd}, lr: {l}, dropout: {do}"
                                )
                                log.info(
                                    f"model_name: {config['model_params']['model_name']}"
                                )
                                log.info(f"new_config_file")
                                log.info(config)

                                train(config)
