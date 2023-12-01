from gensim import models
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import tqdm
import pickle
import pandas as pd
from unidecode import unidecode
import re

batch_size = 128
device = "cuda:0"

# Defining the model
# coder_eds
model_checkpoint = "/export/home/cse200093/Jacques_Bio/data_bio/coder_output/model_150000.pth"
tokenizer_path = "/export/home/cse200093/word-embedding/finetuning-camembert-2021-07-29"
model = torch.load(model_checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# UMLS path
UMLS_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_umls_synonyms/bio_str_SNOMEDCT_US.json"

# Defining save paths
DES_SAVE_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_normalized_des_coder_eds.pkl"
LABEL_SAVE_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_normalized_label_coder_eds.pkl"
EMBEDDINGS_SAVE_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_normalized_embeddings_coder_eds.pt"

# Method to generate embeddings
def get_bert_embed(
    phrase_list, m, tok, normalize=True, summary_method="CLS", tqdm_bar=False
):
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(
            tok.encode_plus(
                phrase,
                max_length=32,
                add_special_tokens=True,
                truncation=True,
                pad_to_max_length=True,
            )["input_ids"]
        )
    m.eval()

    count = len(input_ids)
    now_count = 0
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm.tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(
                input_ids[now_count : min(now_count + batch_size, count)]
            ).to(device)
            if summary_method == "CLS":
                embed = m.bert(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m.bert(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(embed, p=2, dim=1, keepdim=True).clamp(
                    min=1e-12
                )
                embed = embed / embed_norm
            if now_count == 0:
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
        if tqdm_bar:
            pbar.close()
    return output


# Normalisation of words
def normalize(txt):
    return unidecode(
        txt.lower()
        .replace("-", " ")
        .replace("ag ", "antigene ")
        .replace("ac ", "anticorps ")
        .replace("antigenes ", "antigene ")
    )


umls_raw = pd.read_json(UMLS_DIR)
umls_raw["STR"] = umls_raw["STR"].apply(normalize)

words_to_remove = [
    "for",
    "assay",
    "by",
    "tests",
    "minute",
    "exam",
    "with",
    "human",
    "moyenne",
    "in",
    "to",
    "from",
    "analyse",
    "test",
    "level",
    "fluid",
    "laboratory",
    "determination",
    "examination",
    "releasing",
    "quantitative",
    "screening",
    "and",
    "exploration",
    "factor",
    "method",
    "analysis",
    "laboratoire",
    "specimen",
    "or",
    "typing",
    "of",
    "concentration",
    "measurement",
    "detection",
    "procedure",
    "identification",
    "numeration",
    "hour",
    "retired",
    "technique",
    "count",
]

# Second step of normalization: we remove stop words and special characters with regex
regex_words_to_remove = r"\b(?:" + "|".join(words_to_remove) + r")\b"
regex_remove_special_characters = "[^a-zA-Z0-9\s]"

umls_raw["STR"] = umls_raw["STR"].apply(
    lambda syn: re.compile(regex_words_to_remove).sub("", syn)
)
umls_raw["STR"] = umls_raw["STR"].apply(
    lambda syn: re.compile(regex_remove_special_characters).sub("", syn)
)
umls_raw["STR"] = umls_raw["STR"].apply(lambda syn: re.sub(" +", " ", syn).strip())

umls_raw = (
    umls_raw.loc[(~umls_raw["STR"].str.isnumeric()) & (umls_raw["STR"] != "")]
    .groupby(["STR"])
    .agg({"CUI": set, "STR": "first"})
    .reset_index(drop=True)
)
umls_label = umls_raw["CUI"]
umls_des = umls_raw["STR"]

print("Starting embeddings generation...")
umls_embedding = get_bert_embed(umls_des, model, tokenizer, tqdm_bar=True)

# Save embeddings
torch.save(umls_embedding, EMBEDDINGS_SAVE_DIR)

# Save umls_des
open_file = open(DES_SAVE_DIR, "wb")
pickle.dump(umls_des, open_file)
open_file.close()

# Save umls_labels
open_file = open(LABEL_SAVE_DIR, "wb")
pickle.dump(umls_label, open_file)
open_file.close()