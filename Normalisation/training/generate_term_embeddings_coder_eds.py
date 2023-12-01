from gensim import models
import os
import sys
sys.path.append("/export/home/cse200093/Jacques_Bio/normalisation/py_files")
from load_umls import UMLS
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import tqdm
import pickle
import pandas as pd
from unidecode import unidecode

batch_size = 128
device = "cuda:0"

# Defining the model
# coder_all
model_checkpoint = "/export/home/cse200093/Jacques_Bio/data_bio/coder_output/model_150000.pth"
tokenizer_path = "/export/home/cse200093/word-embedding/finetuning-camembert-2021-07-29"
model = torch.load(model_checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Defining data paths
DATA_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/json_annotated_normalisation_formatted/annotated_normalisation_formatted_train_umls_snomed.json"
EMBEDDINGS_SAVE_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/data_embeddings_normalized_snomed_coder_eds.pt"

# Method to generate embeddings
def get_bert_embed(phrase_list, m, tok, normalize=True, summary_method="CLS", tqdm_bar=False):
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
    m.eval()

    count = len(input_ids)
    now_count = 0
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm.tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = m.bert(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m.bert(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
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

# Loading our data to match
data_df = pd.read_json(DATA_DIR)
data_df["term"] = data_df["term"].apply(normalize)
# Merge same terms and keep all possible loincs
data_df = (
    data_df.groupby("term")
    .agg({"term": "first", "annotation": set, "source": set})
    .reset_index(drop=True)
)

umls_embedding = get_bert_embed(data_df["term"].tolist(), model, tokenizer, tqdm_bar=False)
torch.save(umls_embedding, EMBEDDINGS_SAVE_DIR)
