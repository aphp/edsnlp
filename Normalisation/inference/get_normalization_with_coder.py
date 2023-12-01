import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

sys.path.append("/export/home/cse200093/scratch/BioMedics/normalisation/training")


class CoderNormalizer:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        device: str = "cuda:0",
    ):
        self.device = device
        try:
            self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            self.model_from_transformers = True
        except:
            self.model = torch.load(model_name_or_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            self.model_from_transformers = False

    def get_bert_embed(
        self,
        phrase_list,
        normalize=True,
        summary_method="CLS",
        tqdm_bar=False,
        coder_batch_size=128,
    ):
        input_ids = []
        for phrase in phrase_list:
            input_ids.append(
                self.tokenizer.encode_plus(
                    phrase,
                    max_length=32,
                    add_special_tokens=True,
                    truncation=True,
                    pad_to_max_length=True,
                )["input_ids"]
            )
        self.model.eval()

        count = len(input_ids)
        now_count = 0
        with torch.no_grad():
            if tqdm_bar:
                pbar = tqdm(total=count)
            while now_count < count:
                input_gpu_0 = torch.LongTensor(
                    input_ids[now_count : min(now_count + coder_batch_size, count)]
                ).to(self.device)
                if summary_method == "CLS":
                    if self.model_from_transformers:
                        embed = self.model(input_gpu_0)[1]
                    else:
                        embed = self.model.bert(input_gpu_0)[1]
                if summary_method == "MEAN":
                    if self.model_from_transformers:
                        embed = torch.mean(self.model(input_gpu_0)[0], dim=1)
                    else:
                        embed = torch.mean(self.model.bert(input_gpu_0)[0], dim=1)
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
                    pbar.update(min(now_count + coder_batch_size, count) - now_count)
                now_count = min(now_count + coder_batch_size, count)
            if tqdm_bar:
                pbar.close()
        return output

    def get_sim_results(
        self, res_embeddings, umls_embeddings, umls_labels, umls_des, split_size=200
    ):
        label_matches = []
        des_matches = []
        print(f"Number of split: {len(torch.split(res_embeddings, split_size))}")
        for split_res_embeddings in torch.split(res_embeddings, split_size):
            sim = torch.matmul(split_res_embeddings, umls_embeddings.t())
            most_similar = torch.max(sim, dim=1)[1].tolist()
            label_matches_split = [umls_labels[idx] for idx in most_similar]
            des_matches_split = [umls_des[idx] for idx in most_similar]
            label_matches.extend(label_matches_split)
            des_matches.extend(des_matches_split)
        return label_matches, des_matches

    def __call__(
        self,
        umls_labels_list,
        umls_des_list,
        data_list,
        save_umls_embeddings_dir=False,
        save_data_embeddings_dir=False,
        normalize=True,
        summary_method="CLS",
        tqdm_bar=False,
        coder_batch_size=128,
    ):
        umls_embeddings = self.get_bert_embed(
            umls_des_list, normalize, summary_method, tqdm_bar, coder_batch_size
        )
        res_embeddings = self.get_bert_embed(
            data_list, normalize, summary_method, tqdm_bar, coder_batch_size
        )
        if save_umls_embeddings_dir:
            torch.save(umls_embeddings, save_umls_embeddings_dir)
        if save_data_embeddings_dir:
            torch.save(res_embeddings, save_data_embeddings_dir)
        return self.get_sim_results(
            res_embeddings, umls_embeddings, umls_labels_list, umls_des_list
        )
