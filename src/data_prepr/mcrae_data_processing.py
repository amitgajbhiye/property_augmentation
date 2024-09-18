#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator
import numpy as np


# In[2]:


def read(path: str) -> List[str]:
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines() if len(l.strip()) > 0]


# In[3]:


def expand(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    
    """Expands r1: c1 c2 c3 into [r1 c1], [r1, c2], [r1, c3]. Returns labels, data.
    Takes df shape (n, d) and returns the combination of every row and column.
    This is n*d entries. The labels are "row/col" and the data is a 2D array of shape
    (n*d, 1) with each value.
    """
    
    rows = [row.strip() for row, _ in df.iterrows()]
    
    cols = df.columns.to_list()
    cols = [x.replace("_", " ").strip() for x in cols]
    
    data = [] 
    
    for row in rows:
        for col in cols:
            # labels.append("{}/{}".format(row, col))
            data.append([row, col])
    
    _ = [x.insert(2, label) for (x, label) in zip(data, df.to_numpy().reshape(-1))]
    
    # print ("data")
    # print (data)
    
    return data


# In[ ]:





# In[4]:


def train_test_df_split(df: pd.DataFrame, train_uid_path: str, test_uid_path: str):
    
    """Helper for task data getters who split a df by index."""
    train_obj_uids = set(read(train_uid_path))
    test_obj_uids = set(read(test_uid_path))

    train_df = df[df.index.isin(train_obj_uids)]
    test_df = df[df.index.isin(test_obj_uids)]

    return train_df, test_df


# In[5]:


def train_test_df_expand(df: pd.DataFrame, train_uid_path: str, test_uid_path: str):

    """Helper for task data getters who split a df by index, then expand the features."""
    # get our object UID splits, and split df
    
    train_df, test_df = train_test_df_split(df, train_uid_path, test_uid_path)
    
    # print ("train_df")
    # print (train_df)

    labelled_train_data = expand(train_df)
    labelled_test_data = expand(test_df)
    
    return labelled_train_data, labelled_test_data


# In[6]:


def get_abstract_objects_properties():
    # read all abstract data. map {-2, -1, 0} -> 0, {1} -> 1.
    df = pd.read_csv("siamese_concept_property/data/evaluation_data/extended_mcrae/abstract.csv", index_col="objectUID")
    
    # for prop in df.columns:
    #     df[prop] = df[prop].apply(lambda x: 0 if x <= 0 else 1)
            

    return train_test_df_expand(
        df,
        "siamese_concept_property/data/evaluation_data/extended_mcrae/abstract-train-object-uids.txt",
        "siamese_concept_property/data/evaluation_data/extended_mcrae/abstract-test-object-uids.txt",
    )


# In[7]:


train_data, test_data = get_abstract_objects_properties()


# In[8]:


train_df = pd.DataFrame(train_data, columns=["concept", "property", "label"])


# In[9]:


train_df.shape


# In[10]:


train_df = train_df.drop(train_df[train_df["label"] == -2].index)
train_df = train_df.drop(train_df[train_df["label"] == -1].index)
train_df = train_df.drop_duplicates(subset = ["concept", "property"])
train_df = train_df.dropna()
train_df = train_df.sample(frac=1)


# In[11]:


train_df.shape


# In[12]:


train_df["label"].value_counts()


# In[13]:


test_df = pd.DataFrame(test_data, columns=["concept", "property", "label"])


# In[14]:


test_df.shape


# In[15]:


test_df = test_df.drop(test_df[test_df["label"] == -2].index)
test_df = test_df.drop(test_df[test_df["label"] == -1].index)
test_df = test_df.drop_duplicates(subset = ["concept", "property"])
test_df = test_df.dropna()
test_df = test_df.sample(frac=1)


# In[16]:


test_df.shape


# In[17]:


train_file_path = "siamese_concept_property/data/evaluation_data/extended_mcrae/train_mcrae.tsv"
train_df.to_csv(train_file_path, sep="\t", index=False, index_label=None, header=None)


# In[18]:


test_file_path = "siamese_concept_property/data/evaluation_data/extended_mcrae/test_mcrae.tsv"
test_df.to_csv(test_file_path, sep="\t", index=False, index_label=None, header=None)


# In[ ]:





# In[ ]:





# ## Processing of Yixiao Data

# def preprocess_mcrae(pos_file_name, new_file_name):
#     
#     pos_data_df = pd.read_csv(pos_file_name, sep="\t", header=None, names=["property", "concepts"])
#     pos_data_df["label"] = 1
#     
#     neg_data_df = pd.read_csv(new_file_name, sep="\t", header=None, names=["property", "concepts"])
#     neg_data_df["label"] = 0
#     
#     data_df = pd.concat([pos_data_df, neg_data_df], axis=0, ignore_index=True)
#     
#     all_data_list = []
#         
#     for idx in data_df.index:
#         row = data_df.iloc[idx]
#         
#         concept_list = row["concepts"].split(",")
#         prop = row["property"].replace("_", " ").strip()
#         label = row["label"]
#         
#         all_data_list.extend([[c.strip(), prop, label] for c in concept_list])
#           
#     all_data_df = pd.DataFrame(all_data_list, columns=["concept", "property", "label"]).sample(frac=1)
#     
#     print ("all_data_df before dropping duplicates")
#     print (all_data_df.shape)
#     
#     print ("all_data_df after dropping duplicates")
#     
#     duplicated_df = all_data_df.loc[all_data_df.duplicated(subset = ["concept", "property"], keep=False)]
#     print ("duplicated_df.shape :", duplicated_df.shape)
#     
#     # all_data_df = all_data_df.drop_duplicates(subset=["concept", "property"])
#     # all_data_df = all_data_df.dropna()
#     
#     print (all_data_df.shape)
#     
#     print ()
#     
#     print ("all_data_df")
#     print (all_data_df)
#     
#     return all_data_df
# 

# 

# pos_train_file = "siamese_concept_property/data/evaluation_data/MC/pos_train_data.txt"
# neg_train_file = "siamese_concept_property/data/evaluation_data/MC/neg_train_data.txt"
# 
# train_data_df = preprocess_mcrae(pos_train_file, neg_train_file)
# 
# pos_neg_file = "siamese_concept_property/data/evaluation_data/MC/train_pos_neg_mcrae.tsv"
# train_data_df.to_csv(pos_neg_file, sep='\t', index=None, header=None)
# 
# print (train_data_df["label"].value_counts())
# # print ("train_data_df :", train_data_df["concept"].unique())
# print ("train_data_df :", len(train_data_df["concept"].unique()))
# 

# 

# pos_valid_file = "siamese_concept_property/data/evaluation_data/MC/pos_valid_data.txt"
# neg_valid_file = "siamese_concept_property/data/evaluation_data/MC/neg_valid_data.txt"
# 
# valid_data_df = preprocess_mcrae(pos_valid_file, neg_valid_file)
# 
# pos_neg_file = "siamese_concept_property/data/evaluation_data/MC/valid_pos_neg_mcrae.tsv"
# 
# valid_data_df.to_csv(pos_neg_file, sep='\t', index=None, header=None)
# 
# valid_data_df["label"].value_counts()
# 
# print ("valid_data_df :", len(valid_data_df["concept"].unique()))
# 

# df = train_data_df.merge(valid_data_df, how = 'inner', on = ["concept", "property"], indicator=False)

# df

# df_valid = train_data_df.merge(valid_data_df, how = 'inner', on = ["concept"], indicator=False)

# len(df_valid["concept"].unique())

# pos_valid_file = "siamese_concept_property/data/evaluation_data/MC/pos_test_data.txt"
# neg_valid_file = "siamese_concept_property/data/evaluation_data/MC/neg_test_data.txt"
# 
# test_data_df = preprocess_mcrae(pos_valid_file, neg_valid_file)
# 
# pos_neg_file = "siamese_concept_property/data/evaluation_data/MC/test_pos_neg_mcrae.tsv"
# 
# test_data_df.to_csv(pos_neg_file, sep='\t', index=None, header=None)
# 
# test_data_df["label"].value_counts()
# 
# print ("test_data_df :", len(test_data_df["concept"].unique()))
# 
# 

# In[ ]:




