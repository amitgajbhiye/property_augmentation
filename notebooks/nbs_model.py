#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import math

from torch import nn
from torch.nn.functional import normalize

from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np

import pandas as pd
import random

from transformers import BertModel,BertTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# ## Dot Product Model

# In[ ]:



class ConceptPropertyModel(nn.Module):
    def __init__(self):
        super(ConceptPropertyModel, self).__init__()

        # self._concept_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self._property_encoder = BertModel.from_pretrained("bert-base-uncased")

        self._concept_encoder = BertModel.from_pretrained("/scratch/c.scmag3/conceptEmbeddingModel/bertBaseUncasedPreTrained")
        self._property_encoder = BertModel.from_pretrained("/scratch/c.scmag3/conceptEmbeddingModel/bertBaseUncasedPreTrained")

        self.dropout_prob = 0.2
        self.strategy = "mean"

    def forward(
        self,
        concept_input_id,
        concept_attention_mask,
        property_input_id,
        property_attention_mask,
    ):

        concept_output = self._concept_encoder(
            input_ids=concept_input_id, attention_mask=concept_attention_mask
        )

        property_output = self._property_encoder(
            input_ids=property_input_id, attention_mask=property_attention_mask
        )

        concept_last_hidden_states, concept_cls = (
            concept_output.get("last_hidden_state"),
            concept_output.get("pooler_output"),
        )

        property_last_hidden_states, property_cls = (
            property_output.get("last_hidden_state"),
            property_output.get("pooler_output"),
        )

        if self.strategy == "mean":

            # The dot product of the average of the last hidden states of the concept and property hidden states.

            v_concept_avg = torch.sum(
                concept_last_hidden_states
                * concept_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(concept_attention_mask, dim=1, keepdim=True)

            # Normalising concept vectors
            v_concept_avg = normalize(v_concept_avg, p=2, dim=1)

            v_property_avg = torch.sum(
                property_last_hidden_states
                * property_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(property_attention_mask, dim=1, keepdim=True)

            logits = (
                (v_concept_avg * v_property_avg)
                .sum(-1)
                .reshape(v_concept_avg.shape[0], 1)
            )

            return v_concept_avg, v_property_avg, logits

        elif self.strategy == "cls":
            # The dot product of concept property cls vectors.

            # Normalising concept vectors
            concept_cls = normalize(concept_cls, p=2, dim=1)

            logits = (concept_cls * property_cls).sum(-1).reshape(concept_cls.shape[0], 1)

            return concept_cls, property_cls, logits


# ## Data Files

# In[ ]:


file_train = "mscg_train_pos.tsv"
file_valid = "mscg_valid_pos.tsv"

# file_train = "mscg_test_pos.tsv"
# file_valid = "mscg_test_pos.tsv"

context_num = 3
best_model_path = "3_cntx_best_model.pt"


num_epoch = 100
bs = 32
early_stopping_patience = 15


# ## Datasets Class to load data

# In[ ]:


class ConceptPropertyDataset(Dataset):
    
    def __init__(self, data_file_path):

        self.data_df = pd.read_csv(data_file_path,
            sep="\t",
            header=None,
            names=["concept", "property"],
        )
        
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("/scratch/c.scmag3/conceptEmbeddingModel/bertBaseUncasedPreTrained/tokenizer")

        # self.concepts_unique = self.data_df["concept"].unique()
        # self.properties_unique = self.data_df["property"].unique()

        self.concept2idx, self.idx2concept = self.create_concept_idx_dicts()
        self.property2idx, self.idx2property = self.create_property_idx_dicts()

        self.con_pro_dict, self.prop_con_dict = self.populate_dict()
        
        # print ("self.con_pro_dict :", self.con_pro_dict)
        
        self.context_num = context_num
        
    def create_concept_idx_dicts(self):
        
        unique_concepts = self.data_df["concept"].unique()
        
        item2idx, idx2item = {}, {}
        
        for idx, item in enumerate(unique_concepts):
            item2idx[item] = idx
            idx2item[idx] = item

        return item2idx, idx2item

    def create_property_idx_dicts(self):
        
        unique_properties = self.data_df["property"].unique()
        
        item2idx, idx2item = {}, {}
        
        for idx, item in enumerate(unique_properties):
            item2idx[item] = idx
            idx2item[idx] = item

        return item2idx, idx2item

    def populate_dict(self):
        
        concept_property_dict, property_concept_dict = {}, {}
        
        unique_concepts = self.data_df["concept"].unique()
        unique_properties = self.data_df["property"].unique()
        
        self.data_df.set_index("concept", inplace=True)
        
        for concept in unique_concepts:
            
            concept_id = self.concept2idx[concept]
            
            property_list = self.data_df.loc[concept].values.flatten()
            property_ids = np.asarray([self.property2idx[x] for x in property_list])
            
            concept_property_dict[concept_id] = property_ids
        
        self.data_df.reset_index(inplace=True)
        
        self.data_df.set_index("property", inplace=True)
        
        for prop in unique_properties:
            
            property_id = self.property2idx[prop]
            
            concept_list = self.data_df.loc[prop].values.flatten()
            concept_ids = np.asarray([self.concept2idx[x] for x in concept_list])
            
            property_concept_dict[property_id] = concept_ids
        
        self.data_df.reset_index(inplace=True)
        
        return concept_property_dict, property_concept_dict
            
    
    def __len__(self):

        return len(self.data_df)
    
    def __getitem__(self, idx):
        
        return self.data_df["concept"][idx], self.data_df["property"][idx]
    
    def add_context(self, batch):
        
        if self.context_num == 4:
            
            concept_context = "Yesterday, I saw another "
            property_context = "Yesterday, I saw a thing which is "
            
            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]
            
        elif self.context_num == 1:
            
            concept_context = "Concept : "
            property_context = "Property : "
            
            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]
                    
        elif self.context_num == 2:
            
            concept_context = "The notion we are modelling : "
            property_context = "The notion we are modelling : "
            
            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]
            
            
        elif self.context_num == 3:
            
            prefix_num = 5
            suffix_num = 4  
            
            concepts_batch = ["[MASK] " * prefix_num + concept + " " + "[MASK] " * suffix_num + "." for concept in batch[0]]
            property_batch = ["[MASK] " * prefix_num + prop + " " + "[MASK] " * suffix_num + "." for prop in batch[1]]
            
        
        elif self.context_num == 5:
            
            concept_context = "The notion we are modelling is called "
            property_context = "The notion we are modelling is called "
            
            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]
                    
        
        return concepts_batch, property_batch
        
    def tokenize(self, concept_batch, property_batch, concept_max_len=64, property_max_len=64):
        
        concept_ids = self.tokenizer(
            concept_batch,
            max_length=concept_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        property_ids = self.tokenizer(
            property_batch,
            max_length=property_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
                
        return {
            "concept_inp_id": concept_ids.get("input_ids"),
            "concept_atten_mask": concept_ids.get("attention_mask"),
            "property_inp_id": property_ids.get("input_ids"),
            "property_atten_mask": property_ids.get("attention_mask")}
        


# ## Data Loaders

# In[ ]:



train_dataset = ConceptPropertyDataset(file_train)
train_data_sampler = RandomSampler(train_dataset)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=bs, sampler=train_data_sampler)

valid_dataset = ConceptPropertyDataset(file_valid)
valid_sampler = RandomSampler(valid_dataset)
valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, sampler=valid_sampler)


# ## Model

# In[ ]:


model = ConceptPropertyModel()
model.to(device)


# In[ ]:


loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-6)

# warmup_steps = math.ceil(len(train_dl) * num_epoch * 0.1)  # 10% of train data for warm-up
warmup_steps = 0

total_training_steps = (
        len(train_dl) * num_epoch
    )

scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )


# In[ ]:


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


# ## Loss Calculation with in-batch negative sampling

# In[ ]:


def calculate_loss(dataset, batch, concept_embedding, property_embedding, logits, device=device):
    
    # self.concept2idx, self.idx2concept = self.create_concept_idx_dicts()
    # self.property2idx, self.idx2property = self.create_property_idx_dicts()
    
    # print ("con_pro_dict :", dataset.con_pro_dict, "\n")
    
    num_neg_concept = random.randint(0, concept_embedding.shape[0])
    
    # print ("\t  num_neg_concept :", num_neg_concept, flush=True)
    
    batch_logits, batch_labels = [], []
    
    concept_id_list_for_batch = torch.tensor([dataset.concept2idx[concept] for concept in batch[0]], device=device)
    property_id_list_for_batch = torch.tensor([dataset.property2idx[prop] for prop in batch[1]], device=device)
    
    # print ("concept_id_list_for_batch :", concept_id_list_for_batch)
    # print ("property_id_list_for_batch :", property_id_list_for_batch)
        
    # neg_concept_list, neg_property_list = [], []
    
    logits_pos_concepts = ((concept_embedding * property_embedding).sum(-1).reshape(concept_embedding.shape[0], 1))
    labels_pos_concepts = torch.ones_like(logits_pos_concepts, dtype=torch.float32, device=device)
    
    batch_logits.append(logits_pos_concepts.flatten())
    batch_labels.append(labels_pos_concepts.flatten())
    
    # print ("\nlogits_pos_concepts :", logits_pos_concepts)
    # print ("labels :", labels)
    
    loss_pos_concept = loss_fn(logits_pos_concepts, labels_pos_concepts)
    # print ("Loss positive concepts :", loss_pos_concept)
    
    loss_neg_concept = 0.0
    loss_neg_property = 0.0 
    
    for i in range(len(concept_id_list_for_batch)):
        
        if i < num_neg_concept:
            
            concept_id = concept_id_list_for_batch[i]
            
            # Extracting the property of the concept at the whole dataset level.
            property_id_list_for_concept = torch.tensor(dataset.con_pro_dict[concept_id.item()], device=device)

            # Extracting the negative property by excluding the properties that the concept may have at the  whole dataset level
            negative_property_id_for_concept = torch.tensor([x for x in property_id_list_for_batch if x not in property_id_list_for_concept], device=device)

            positive_property_for_concept_mask = torch.tensor([[1] if x in negative_property_id_for_concept else [0] for x in property_id_list_for_batch], device=device)

            neg_property_embedding = torch.mul(property_embedding, positive_property_for_concept_mask)

            concept_i_repeated = concept_embedding[i].unsqueeze(0).repeat(concept_embedding.shape[0], 1)

            logits_neg_concepts = ((concept_i_repeated * neg_property_embedding).sum(-1).reshape(concept_i_repeated.shape[0], 1))

            labels_neg_concepts = torch.zeros_like(logits_neg_concepts, dtype=torch.float32, device=device)

            batch_logits.append(logits_neg_concepts.flatten())
            batch_labels.append(labels_neg_concepts.flatten())
            
            loss_neg_concept += loss_fn(logits_neg_concepts, labels_neg_concepts)
            
            # print (loss_neg_concept)
        
        else:
            
            property_id = property_id_list_for_batch[i]
            
            # Extracting the concept for the property at the whole dataset level
            concept_id_list_for_property = torch.tensor(dataset.prop_con_dict[property_id.item()], device=device)
            
            # Extracting the negative concepts for the property by excluding the concepts that the property may have at the whole dataset level.
            negative_concept_id_for_property = torch.tensor([x for x in concept_id_list_for_batch if x not in concept_id_list_for_property], device=device)
            
            # [f(x) if condition else g(x) for x in sequence]
            # mask i.e zero out the postive concept indices in concept_id_list_for_batch for this particular property id.
            positive_concept_for_property_mask = torch.tensor([[1] if x in negative_concept_id_for_property else [0] for x in concept_id_list_for_batch], device=device)
            
            neg_concept_embedding = torch.mul(concept_embedding, positive_concept_for_property_mask)
            
            property_i_repeated = property_embedding[i].unsqueeze(0).repeat(property_embedding.shape[0], 1)
            
            logits_neg_property = ((neg_concept_embedding * property_i_repeated).sum(-1).reshape(neg_concept_embedding.shape[0], 1))
            
            labels_neg_property = torch.zeros_like(logits_neg_property, dtype=torch.float32, device=device)
            
            batch_logits.append(logits_neg_property.flatten())
            batch_labels.append(labels_neg_property.flatten())
            
            loss_neg_property += loss_fn(logits_neg_property, labels_neg_property)
    
    
#     print ("\t loss_pos_concept :", loss_pos_concept, flush=True)
#     print ("\t loss_neg_concept :", loss_neg_concept, flush=True)
#     print ("\t loss_neg_property :", loss_neg_property, flush=True)
    
#     print ("\t Total Batch loss : ", loss_pos_concept + loss_neg_concept + loss_neg_property, flush=True)
#     print ()
#     print ("batch_logits :", batch_logits)
#     print ("batch_labels :", batch_labels)
    
    batch_logits = torch.vstack(batch_logits).reshape(-1, 1)
    batch_labels = torch.vstack(batch_labels).reshape(-1, 1)
        
    return loss_pos_concept + loss_neg_concept + loss_neg_property, batch_logits, batch_labels
    


# ## Train Function

# In[ ]:


def train():
    
    epoch_loss = 0.0 
    
    model.train()
    for step, batch in enumerate(train_dl):
        
        model.zero_grad()
        
        concepts_batch, property_batch = train_dataset.add_context(batch)
        
        ids_dict = train_dataset.tokenize(concepts_batch, property_batch)

        concept_inp_id, concept_attention_mask, property_input_id, property_attention_mask = [val.to(device) for _, val in ids_dict.items()]

        concept_embedding, property_embedding, logits =  model(concept_input_id=concept_inp_id,
                                                               concept_attention_mask=concept_attention_mask,
                                                               property_input_id=property_input_id,
                                                               property_attention_mask=property_attention_mask)

        batch_loss, batch_logits, batch_labels = calculate_loss(train_dataset, batch, concept_embedding, property_embedding, logits, device=device)
        
        epoch_loss += batch_loss.item()
        
        batch_loss.backward()
        torch.cuda.empty_cache()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
    
    avg_epoch_loss = epoch_loss / len(train_dl)
    
    return avg_epoch_loss
            


# ## Evaluate Function

# In[ ]:


def evaluate():
    
    val_loss = 0.0
    
    model.eval()
    
    epoch_logits, epoch_labels = [], []
    
    for step, batch in enumerate(valid_dl):
        
        concepts_batch, property_batch = valid_dataset.add_context(batch)
        
        ids_dict = valid_dataset.tokenize(concepts_batch, property_batch)

        concept_inp_id, concept_attention_mask, property_input_id, property_attention_mask = [val.to(device) for _, val in ids_dict.items()]
        
        with torch.no_grad():
            
            concept_embedding, property_embedding, logits =  model(concept_input_id=concept_inp_id,
                                                               concept_attention_mask=concept_attention_mask,
                                                               property_input_id=property_input_id,
                                                               property_attention_mask=property_attention_mask)
        
        batch_loss, batch_logits, batch_labels = calculate_loss(valid_dataset, batch, concept_embedding, property_embedding, logits, device=device)
        
        epoch_logits.append(batch_logits)
        epoch_labels.append(batch_labels)
        
        val_loss += batch_loss.item()
        torch.cuda.empty_cache()
    
    epoch_logits = torch.round(torch.sigmoid(torch.vstack(epoch_logits))).reshape(-1, 1).detach().cpu().numpy()
    epoch_labels = torch.vstack(epoch_labels).reshape(-1, 1).detach().cpu().numpy()

    print ("epoch_logits type: ", type(epoch_logits), flush=True)
    print ("epoch_logits shape :", epoch_logits.shape, flush=True)
    
    print ("epoch_labels type :", type(epoch_labels), flush=True)
    print ("epoch_labels type :", epoch_labels.shape, flush=True)
    
    scores = compute_scores(epoch_labels, epoch_logits)
    
    avg_val_loss = val_loss / len(valid_dl)
    
    return avg_val_loss, scores
            


# ## Train Loop

# In[ ]:


best_val_f1 = 0.0
start_epoch = 1

patience_counter = 0

for epoch in range(start_epoch, num_epoch+1):
    
    train_loss = train()
    val_loss, scores = evaluate()
    
    print (f"\n Epoch {epoch}")
    print(f"Train Epoch : {epoch} ----> Train Average Epoch Loss : {train_loss}", flush=True)
    print(f"Validation Epoch : {epoch} ----> Val Average Epoch Loss : {val_loss}", flush=True)
    print (f"Best Validation F1 yet :", best_val_f1)
    
    print ("\nValidation Metrices")
    
    for key, val in scores.items():
        print (f"{key} : {val}", flush=True)
    
    val_f1 = scores.get("binary_f1")
    
    if val_f1 < best_val_f1:
        patience_counter += 1 
        
    else:
        patience_counter = 0
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
    
    if patience_counter > early_stopping_patience:
        break
      


# ## Test data class to load data for without in-batch negative sampling 

# In[ ]:


class TestDataset(Dataset):
    
    def __init__(self, data_file_path):

        self.data_df = pd.read_csv(data_file_path,
            sep="\t",
            header=None,
            names=["concept", "property", "label"],
        )
        
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("/scratch/c.scmag3/conceptEmbeddingModel/bertBaseUncasedPreTrained/tokenizer")

        self.context_num = context_num
        self.label = self.data_df["label"].values
        
    
    def __len__(self):

        return len(self.data_df)
    
    def __getitem__(self, idx):
        
        return self.data_df["concept"][idx], self.data_df["property"][idx]
    
    def add_context(self, batch):
        
        if self.context_num == 4:
            
            concept_context = "Yesterday, I saw another "
            property_context = "Yesterday, I saw a thing which is "
            
            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]
            
        elif self.context_num == 1:
            
            concept_context = "Concept : "
            property_context = "Property : "
            
            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]
                    
        elif self.context_num == 2:
            
            concept_context = "The notion we are modelling : "
            property_context = "The notion we are modelling : "
            
            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]
            
            
        elif self.context_num == 3:
            
            prefix_num = 5
            suffix_num = 4  
            
            concepts_batch = ["[MASK] " * prefix_num + concept + " " + "[MASK] " * suffix_num + "." for concept in batch[0]]
            property_batch = ["[MASK] " * prefix_num + prop + " " + "[MASK] " * suffix_num + "." for prop in batch[1]]
            
        
        elif self.context_num == 5:
            
            concept_context = "The notion we are modelling is called "
            property_context = "The notion we are modelling is called "
            
            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]
                    
        
        return concepts_batch, property_batch
        
    def tokenize(self, concept_batch, property_batch, concept_max_len=64, property_max_len=64):
        
        concept_ids = self.tokenizer(
            concept_batch,
            max_length=concept_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        property_ids = self.tokenizer(
            property_batch,
            max_length=property_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
                
        return {"concept_inp_id": concept_ids.get("input_ids"),
            "concept_atten_mask": concept_ids.get("attention_mask"),
            "property_inp_id": property_ids.get("input_ids"),
            "property_atten_mask": property_ids.get("attention_mask")}
        


# In[ ]:


# Test Data with 5 neg samples per concept-property pair

file_test = "65k_test_ms_concept_graph.tsv"
test_dataset = TestDataset(file_test)
test_sampler = SequentialSampler(test_dataset)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=bs, sampler=test_sampler)


# In[ ]:


def test_old_data(test_dataset, test_dl):
    
    print ("Testing the model with old data 5 negatives")
    
    best_model = best_model_path
    
    model = ConceptPropertyModel()
    model.load_state_dict(torch.load(best_model))
    model.eval()
    model.to(device)
    
    label = test_dataset.label
    all_test_preds = [] 
    
    for step, batch in enumerate(test_dl):
        
        concepts_batch, property_batch = test_dataset.add_context(batch)
        
        ids_dict = test_dataset.tokenize(concepts_batch, property_batch)

        concept_inp_id, concept_attention_mask, property_input_id, property_attention_mask = [val.to(device) for _, val in ids_dict.items()]
        
        with torch.no_grad():
            
            concept_embedding, property_embedding, logits =  model(concept_input_id=concept_inp_id,
                                                               concept_attention_mask=concept_attention_mask,
                                                               property_input_id=property_input_id,
                                                               property_attention_mask=property_attention_mask)
            
        preds = torch.round(torch.sigmoid(logits))
        all_test_preds.extend(preds.detach().cpu().numpy().flatten())
            
    
    scores = compute_scores(label, all_test_preds)
    
    print ("\nTest Metrices")
    for key, val in scores.items():
        print (f"{key} : {val}", flush=True)
    


# In[ ]:


test_old_data(test_dataset, test_dl)


# In[ ]:



# Test Data for in-batch negative sampling

positive_only_test_file = "mscg_test_pos.tsv"

test_nbs_dataset = ConceptPropertyDataset(positive_only_test_file)

test_nbs_data_sampler = SequentialSampler(test_nbs_dataset)

test_nbs_dl = torch.utils.data.DataLoader(test_nbs_dataset, batch_size=bs, sampler=test_nbs_data_sampler)


# In[ ]:


def test_nbs(test_nbs_dataset, test_nbs_dl):
    
    print ()
    print ("*" * 50)
    print ("\nTesting the model with Negative Batch sampling")
    
    best_model = best_model_path
    
    model = ConceptPropertyModel()
    model.load_state_dict(torch.load(best_model))
    
    model.eval()
    model.to(device)
    
    epoch_logits, epoch_labels = [], []
    
    epoch_loss = 0.0
    for step, batch in enumerate(test_nbs_dl):
        
        concepts_batch, property_batch = test_nbs_dataset.add_context(batch)
        
        ids_dict = test_nbs_dataset.tokenize(concepts_batch, property_batch)

        concept_inp_id, concept_attention_mask, property_input_id, property_attention_mask = [val.to(device) for _, val in ids_dict.items()]
        
        with torch.no_grad():
            
            concept_embedding, property_embedding, logits =  model(concept_input_id=concept_inp_id,
                                                               concept_attention_mask=concept_attention_mask,
                                                               property_input_id=property_input_id,
                                                               property_attention_mask=property_attention_mask)
        
        batch_loss, batch_logits, batch_labels = calculate_loss(test_nbs_dataset, batch, concept_embedding, property_embedding, logits, device=device)
        
        epoch_logits.append(batch_logits)
        epoch_labels.append(batch_labels)
        
        epoch_loss += batch_loss.item()
        torch.cuda.empty_cache()
    
    epoch_logits = torch.round(torch.sigmoid(torch.vstack(epoch_logits))).reshape(-1, 1).detach().cpu().numpy()
    epoch_labels = torch.vstack(epoch_labels).reshape(-1, 1).detach().cpu().numpy()

    scores = compute_scores(epoch_labels, epoch_logits)
    
    for key, value in scores.items():
        print (f"{key} : {value}", flush=True)
            


# In[ ]:


test_nbs(test_nbs_dataset, test_nbs_dl)


# In[ ]:




