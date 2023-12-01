from gensim import models
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import tqdm
import pickle
import pandas as pd

device = "cpu"
EMBEDDINGS_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_normalized_embeddings_coder_eds.pt"
RES_EMBEDDINGS_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/data_embeddings_normalized_snomed_coder_eds.pt"
LABEL_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_normalized_label_coder_eds.pkl"
DES_DIR = "/export/home/cse200093/Jacques_Bio/data_bio/normalisation_embeddings/umls_normalized_des_coder_eds.pkl"

# LOAD UMLS EMBEDDINGS ALREADY GENERATED
umls_embeddings = torch.load(EMBEDDINGS_DIR, map_location=torch.device(device))
# LOAD EMBEDDINGS FROM ANNOTATED DATASET ALREADY GENERATED
res_embeddings = torch.load(RES_EMBEDDINGS_DIR, map_location=torch.device(device))

# LOAD CORRESPONDANCE FILE BETWEEN EMBEDDINGS AND CUIS AND DES
with open(LABEL_DIR, "rb") as f:
    umls_labels= pickle.load(f)
    
with open(DES_DIR, "rb") as f:
    umls_des= pickle.load(f)

sim = torch.matmul(res_embeddings, umls_embeddings.t())
most_similar = torch.max(sim, dim=1)[1].tolist()
label_matches = [umls_labels[idx] for idx in most_similar]
des_matches = [umls_des[idx] for idx in most_similar]
print(label_matches)
print(des_matches)