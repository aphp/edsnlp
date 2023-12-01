print("BONJOUR")

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

batch_size = 128
device = "cuda:0"

# Defining the model
# coder_all
model_checkpoint = '/export/home/cse200093/coder_all'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint).to(device)

# Defining save paths
DES_SAVE_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_des.pkl"
LABEL_SAVE_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_label.pkl"
EMBEDDINGS_SAVE_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_embeddings.pt"

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
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
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

def get_umls():
    umls_label = []
    umls_label_set = set()
    umls_des = []
    umls = UMLS("/export/home/cse200093/deep_mlg_normalization/resources/umls/2021AB", lang_range=['ENG', 'FRA'], only_load_dict=True)
    umls.load_sty()
    umls_sty = umls.cui2sty
    for cui in tqdm.tqdm(umls.cui2str):
        if not cui in umls_label_set and umls_sty[cui] == 'Laboratory Procedure':
            tmp_str = list(umls.cui2str[cui])
            umls_label.extend([cui] * len(tmp_str))
            umls_des.extend(tmp_str)
            umls_label_set.update([cui])
    print(len(umls_des))
    return umls_label, umls_des

umls_label, umls_des = get_umls()
umls_embedding = get_bert_embed(umls_des, model, tokenizer, tqdm_bar=False)

"""# Save embeddings
torch.save(umls_embedding, EMBEDDINGS_SAVE_DIR)

# Save umls_des
open_file = open(DES_SAVE_DIR, "wb")
pickle.dump(umls_des, open_file)
open_file.close()

# Save umls_labels
open_file = open(LABEL_SAVE_DIR, "wb")
pickle.dump(umls_label, open_file)
open_file.close()"""