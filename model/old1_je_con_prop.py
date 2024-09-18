import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle


from argparse import ArgumentParser

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"The Model is Trained on : {device}", flush=True)


def compute_scores(labels, preds):

    assert len(labels) == len(
        preds
    ), f"labels len: {len(labels)} is not equal to preds len {len(preds)}"

    scores = {
        "binary_f1": round(f1_score(labels, preds, average="binary"), 4),
        "micro_f1": round(f1_score(labels, preds, average="micro"), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(labels, preds, average="weighted"), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "classification report": classification_report(labels, preds, labels=[0, 1]),
        "confusion matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }

    return scores


#### Parameters ####

# MODEL_CLASS = {
#     "bert-base-seq-classification": (
#         BertForSequenceClassification,
#         "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/tokenizer",
#         "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/model",
#     ),
# }

print(f"Property Conjuction Joint Encoder Model- Step3", flush=True)

hawk_bb_tokenizer = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/tokenizer"
hawk_bb_model = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/model"

data_path = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data"


# train_file = os.path.join(data_path, "5_neg_prop_conj_train_cnet_premium.tsv")
# valid_file = os.path.join(data_path, "5_neg_prop_conj_valid_cnet_premium.tsv")

# train_file = "/scratch/c.scmag3/biencoder_concept_property/trained_models/redo_prop_conj_exp/train_cnetp_5prop_conj.tsv"
# valid_file = "/scratch/c.scmag3/biencoder_concept_property/trained_models/redo_prop_conj_exp/valid_cnetp_5prop_conj.tsv"
# test_file = None

# train_file = os.path.join(data_path, "dummy_prop_conj.tsv")
# valid_file = os.path.join(data_path, "dummy_prop_conj.tsv")

# model_save_path = (
#     "/scratch/c.scmag3/biencoder_concept_property/trained_models/redo_prop_conj_exp/"
# )
# model_name = "je_prop_conj_pretrained_cnetp_je_5neg_cnetp_filtered_props.pt"
# best_model_path = os.path.join(model_save_path, model_name)

# For Fine Tuning
best_model_path = None
model_name = None
train_file = None
valid_file = None


max_len = 64

num_labels = 2
# batch_size = 64
batch_size = 32
# num_epoch = 100
num_epoch = 8
lr = 2e-6

load_pretrained = True
# pretrained_model_path = "/scratch/c.scmag3/biencoder_concept_property/trained_models/joint_encoder_gkbcnet_cnethasprop/joint_encoder_concept_property_gkbcnet_cnethasprop_step2_pretrained_model.pt"
# pretrained_model_path = "/scratch/c.scmag3/biencoder_concept_property/trained_models/joint_encoder_gkbcnet_cnethasprop/je_con_prop_cnet_premium_10negdata_pretrained_model.pt"
pretrained_model_path = "/scratch/c.scmag3/biencoder_concept_property/trained_models/joint_encoder_gkbcnet_cnethasprop/je_con_prop_cnet_premium_20negdata_pretrained_model.pt"

print(flush=True)
print(f"Train File : {train_file}", flush=True)
print(f"Valid File : {valid_file}", flush=True)
print(f"Load Pretrained : {load_pretrained}")
print(f"Pretrained Model Path : {pretrained_model_path}")

print(f"Batch Size : {batch_size}")
print(f"Num Epoch : {num_epoch}")
print(f"Learning Rate : {lr}")
print(flush=True)


class DatasetConceptProperty(Dataset):
    def __init__(self, concept_property_file, max_len=max_len):

        if isinstance(concept_property_file, pd.DataFrame):
            self.data_df = concept_property_file

        elif os.path.isfile(concept_property_file):

            self.data_df = pd.read_csv(
                concept_property_file,
                sep="\t",
                header=None,
                names=["concept", "property", "label"],
            )

        print(f"Loaded Dataframe")
        print(f"{self.data_df.head(n=20)}")

        self.tokenizer = BertTokenizer.from_pretrained(hawk_bb_tokenizer)
        self.max_len = max_len

        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token

        # self.labels = torch.tensor(self.data_df["labels"].values)

    def __len__(self):

        return len(self.data_df)

    def __getitem__(self, idx):

        concept = self.data_df["concept"][idx].replace(".", "").strip()
        property = self.data_df["property"][idx].replace(".", "").strip()
        labels = self.data_df["label"][idx]

        # print(f"{con_prop_conj} - {prop_to_predict} - {labels.item()}", flush=True)

        encoded_dict = self.tokenizer.encode_plus(
            text=concept,
            text_pair=property,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        input_ids = encoded_dict["input_ids"]
        attention_mask = encoded_dict["attention_mask"]
        token_type_ids = encoded_dict["token_type_ids"]

        # print(f"input_ids : {input_ids}")
        # print(f"attention_mask : {attention_mask}")
        # print(f"token_type_ids : {token_type_ids}")
        # print(f"labels :", {labels})
        # print()

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ModelConceptProperty(nn.Module):
    def __init__(self):
        super(ModelConceptProperty, self).__init__()

        # self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(
            hawk_bb_model, num_labels=num_labels
        )

        assert self.bert.config.num_labels == 2

    def forward(self, input_ids, token_type_ids, attention_mask, labels):

        output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss, logits = output.loss, output.logits

        return loss, logits


def load_pretrained_model(pretrained_model_path):

    model = ModelConceptProperty()
    model.load_state_dict(torch.load(pretrained_model_path))

    print(f"The pretrained model is loaded from : {pretrained_model_path}", flush=True)

    return model


def prepare_data_and_models(
    train_file, valid_file=None, test_file=None, load_pretrained=False
):

    train_data = DatasetConceptProperty(train_file)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=default_convert,
    )

    print(f"Train Data DF shape : {train_data.data_df.shape}")

    if valid_file is not None:
        val_data = DatasetConceptProperty(valid_file)
        val_sampler = RandomSampler(val_data)
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=default_convert,
        )
        print(f"Valid Data DF shape : {val_data.data_df.shape}")
    else:
        val_dataloader = None

    if test_file is not None:
        test_data = DatasetConceptProperty(test_file)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=default_convert,
        )
        print(f"Test Data DF shape : {test_data.data_df.shape}")
    else:
        test_dataloader = None

    if load_pretrained:
        print(f"load_pretrained : {load_pretrained}")
        model = load_pretrained_model(pretrained_model_path)
    else:
        model = ModelConceptProperty()

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * num_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    return (
        model,
        scheduler,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )


def train_on_single_epoch(model, scheduler, optimizer, train_dataloader):

    total_epoch_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        model.zero_grad()

        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
        token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0).to(
            device
        )
        attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0).to(
            device
        )
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        loss, logits = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 and not step == 0:
            print(
                "   Batch {} of Batch {} ---> Batch Loss {}".format(
                    step, len(train_dataloader), round(loss.item(), 4)
                ),
                flush=True,
            )

    avg_train_loss = total_epoch_loss / len(train_dataloader)

    print("Average Train Loss :", round(avg_train_loss, 4), flush=True)

    return avg_train_loss, model


def evaluate(model, dataloader):

    model.eval()

    val_loss, val_accuracy, val_preds, val_labels = [], [], [], []

    for step, batch in enumerate(dataloader):

        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
        token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0).to(
            device
        )
        attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0).to(
            device
        )
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        val_loss.append(loss.item())

        batch_preds = torch.argmax(logits, dim=1).flatten()
        batch_accuracy = (labels == batch_preds).cpu().numpy().mean() * 100

        val_accuracy.append(batch_accuracy)
        val_preds.extend(batch_preds.cpu().detach().numpy())
        val_labels.extend(labels.cpu().detach().numpy())

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    val_preds = np.array(val_preds).flatten()
    val_labels = np.array(val_labels).flatten()

    return val_loss, val_preds, val_labels


def train(
    model,
    scheduler,
    optimizer,
    train_dataloader,
    val_dataloader=None,
    test_dataloader=None,
):

    best_valid_loss, best_valid_f1 = 0.0, 0.0

    patience_early_stopping = 10
    patience_counter = 0
    start_epoch = 1

    train_losses, valid_losses = [], []

    for epoch in range(start_epoch, num_epoch + 1):

        print("\n Epoch {:} of {:}".format(epoch, num_epoch), flush=True)

        train_loss, model = train_on_single_epoch(
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
        )

        if val_dataloader is not None:

            print(f"Running Validation ....", flush=True)

            valid_loss, valid_preds, valid_gold_labels = evaluate(
                model=model, dataloader=val_dataloader
            )

            scores = compute_scores(valid_gold_labels, valid_preds)
            valid_binary_f1 = scores["binary_f1"]

            if valid_binary_f1 < best_valid_f1:
                patience_counter += 1
            else:
                patience_counter = 0
                best_valid_f1 = valid_binary_f1

                print("\n", "+" * 20, flush=True)
                print("Saving Best Model at Epoch :", epoch, model_name, flush=True)
                print("Epoch :", epoch, flush=True)
                print("   Best Validation F1:", best_valid_f1, flush=True)

                torch.save(model.state_dict(), best_model_path)

                print(f"The best model is saved at : {best_model_path}", flush=True)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print("\n", flush=True)
            print("+" * 50, flush=True)

            print("valid_preds shape:", valid_preds.shape, flush=True)
            print("val_gold_labels shape:", valid_gold_labels.shape, flush=True)

            print(f"\nTraining Loss: {round(train_loss, 4)}", flush=True)
            print(f"Validation Loss: {round(valid_loss, 4)}", flush=True)

            print(f"Current Validation F1 Score Binary {valid_binary_f1}", flush=True)
            print(f"Best Validation F1 Score Yet : {best_valid_f1}", flush=True)

            print("Validation Scores")
            for key, value in scores.items():
                print(f" {key} :  {value}", flush=True)

            if patience_counter > patience_early_stopping:

                print(f"\nTrain Losses :", train_losses, flush=True)
                print(f"Validation Losses: ", valid_losses, flush=True)

                print("Early Stopping ---> Patience Reached!!!", flush=True)
                break

    if test_dataloader is not None:
        print(f"Testing the Model on Fold ....", flush=True)

        _, test_preds, test_gold_labels = evaluate(
            model=model, dataloader=test_dataloader
        )

        scores = compute_scores(test_gold_labels, test_preds)

        print(f"Fold test_gold_labels.shape : {test_gold_labels.shape}", flush=True)
        print(f"Fold test_preds.shape : {test_preds.shape}", flush=True)

        assert (
            test_gold_labels.shape == test_preds.shape
        ), "shape of fold's labels not equal to fold's preds"

        print("************ Test Scores ************", flush=True)
        for key, value in scores.items():
            print(f" {key} :  {value}", flush=True)

        return test_preds, test_gold_labels


################ Fine Tuning Code Starts Here ################


def concept_split_training(train_file, test_file, load_pretrained):

    print(f"Training the Model on Concept Split", flush=True)
    print(f"Train File : {train_file}", flush=True)
    print(f"Test File : {test_file}", flush=True)
    print(f"Load Pretrained :{load_pretrained}", flush=True)

    (
        model,
        scheduler,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = prepare_data_and_models(
        train_file=train_file,
        valid_file=None,
        test_file=test_file,
        load_pretrained=load_pretrained,
    )

    train(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
    )


def do_cv(cv_type):

    if cv_type == "concept_split":
        pass
        # concept_split_training(train_file, test_file, load_pretrained)

    elif cv_type in ("property_split", "concept_property_split"):

        if cv_type == "property_split":

            num_fold = 5
            dir_name = "/scratch/c.scmag3/biencoder_concept_property/data/evaluation_data/mcrae_prop_split_train_test_files"
            train_file_base_name = "train_prop_split_con_prop.pkl"
            test_file_base_name = "test_prop_split_con_prop.pkl"

            print(f"CV Type : {cv_type}", flush=True)
            print(f"Training the Property Split", flush=True)
            print(f"Number of Folds: {num_fold}", flush=True)

        elif cv_type == "concept_property_split":

            num_fold = 9
            dir_name = "data/evaluation_data/mcrae_con_prop_split_train_test_files"
            train_file_base_name = "train_con_prop_split_con_prop.pkl"
            test_file_base_name = "test_con_prop_split_con_prop.pkl"

            print(f"Training the Concept Property Split", flush=True)
            print(f"Number of Folds: {num_fold}", flush=True)

        else:
            raise Exception(f"Specify a correct Split")

        all_folds_test_preds, all_folds_test_labels = [], []
        for fold in range(num_fold):

            print(flush=True)
            print("*" * 50)
            print(f"Training the model on Fold : {fold}/{num_fold}", flush=True)
            print("*" * 50, flush=True)
            print(flush=True)

            train_file_name = os.path.join(dir_name, f"{fold}_{train_file_base_name}")
            test_file_name = os.path.join(dir_name, f"{fold}_{test_file_base_name}")

            print(f"Train File Name : {train_file_name}", flush=True)
            print(f"Test File Name : {test_file_name}", flush=True)
            print(flush=True)

            with open(train_file_name, "rb") as train_pkl, open(
                test_file_name, "rb"
            ) as test_pkl:

                train_df = pickle.load(train_pkl)
                test_df = pickle.load(test_pkl)

            (
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = prepare_data_and_models(
                train_file=train_df,
                valid_file=None,
                test_file=test_df,
                load_pretrained=load_pretrained,
            )

            fold_test_preds, fold_test_gold_labels = train(
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            )

            all_folds_test_preds.extend(fold_test_preds)
            all_folds_test_labels.extend(fold_test_gold_labels)

        all_folds_test_preds = np.array(all_folds_test_preds).flatten()
        all_folds_test_labels = np.array(all_folds_test_labels).flatten()

        print(flush=True)
        print(f"Shape of All Folds Preds : {all_folds_test_preds.shape}", flush=True)
        print(f"Shape of All Folds Labels : {all_folds_test_labels.shape}", flush=True)

        assert (
            all_folds_test_preds.shape == all_folds_test_labels.shape
        ), "shape of all folds labels not equal to all folds preds"

        print(flush=True)
        print("*" * 50, flush=True)
        print(f"Calculating the scores for All Folds", flush=True)
        scores = compute_scores(all_folds_test_labels, all_folds_test_preds)

        for key, value in scores.items():
            print(f"{key} : {value}", flush=True)
        print("*" * 50, flush=True)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Joint Encoder Property Conjuction Model - Step 3"
    )

    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--cv_type",)

    args = parser.parse_args()

    print(f"Supplied Arguments", flush=True)
    print("args.pretrain :", args.pretrain, flush=True)
    print("args.finetune:", args.finetune, flush=True)

    if args.pretrain:

        (
            model,
            scheduler,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = prepare_data_and_models(
            train_file=train_file,
            valid_file=valid_file,
            test_file=None,
            load_pretrained=load_pretrained,
        )

        train(
            model,
            scheduler,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
        )

    elif args.finetune:

        print(f"Arg Finetune : {args.finetune}", flush=True)
        print(f"Arg CV Type : {args.cv_type}", flush=True)

        cv_type = args.cv_type

        do_cv(cv_type=cv_type)

