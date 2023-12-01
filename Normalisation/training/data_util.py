import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from load_umls import UMLS
from torch.utils.data import Dataset, DataLoader
from random import sample
from sampler_util import FixedLengthBatchSampler, my_collate_fn
from torch.utils.data.sampler import RandomSampler
import ipdb
from time import time
import json
from pathlib import Path


def pad(list_ids, pad_length, pad_mark=0):
    output = []
    for l in list_ids:
        if len(l) > pad_length:
            output.append(l[0:pad_length])
        else:
            output.append(l + [pad_mark] * (pad_length - len(l)))
    return output


def my_sample(lst, lst_length, start, length):
    start = start % lst_length
    if start + length < lst_length:
        return lst[start:start + length]
    return lst[start:] + lst[0:start + length - lst_length]


class UMLSDataset(Dataset):
    def __init__(self, umls_folder, model_name_or_path, lang, json_save_path=None, max_lui_per_cui=8, max_length=32):
        self.umls = UMLS(umls_folder, lang_range=lang)
        self.len = len(self.umls.rel)
        self.max_lui_per_cui = max_lui_per_cui
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("/export/home/cse200093/scratch/word-embedding/finetuning-camembert-2021-07-29")
        self.json_save_path = json_save_path
        self.calculate_class_count()

    def calculate_class_count(self):
        print("Calculate class count")

        self.cui2id = {cui: index for index,
                       cui in enumerate(self.umls.cui2str.keys())}

        self.re_set = set()
        self.rel_set = set()
        for r in self.umls.rel:
            _, _, re, rel = r.split("\t")
            self.re_set.update([re])
            self.rel_set.update([rel])
        self.re_set = list(self.re_set)
        self.rel_set = list(self.rel_set)
        self.re_set.sort()
        self.rel_set.sort()

        self.re2id = {re: index for index, re in enumerate(self.re_set)}
        self.rel2id = {rel: index for index, rel in enumerate(self.rel_set)}

        sty_list = list(set(self.umls.cui2sty.values()))
        sty_list.sort()
        self.sty2id = {sty: index for index, sty in enumerate(sty_list)}

        if self.json_save_path:
            with open(os.path.join(self.json_save_path, "re2id.json"), "w") as f:
                json.dump(self.re2id, f)
            with open(os.path.join(self.json_save_path, "rel2id.json"), "w") as f:
                json.dump(self.rel2id, f)
            with open(os.path.join(self.json_save_path, "sty2id.json"), "w") as f:
                json.dump(self.sty2id, f)

        print("CUI:", len(self.cui2id))
        print("RE:", len(self.re2id))
        print("REL:", len(self.rel2id))
        print("STY:", len(self.sty2id))

    def tokenize_one(self, string):
        return self.tokenizer.encode_plus(string, max_length=self.max_length, truncation=True)['input_ids']

    # @profile
    def __getitem__(self, index):
        cui0, cui1, re, rel = self.umls.rel[index].split("\t")

        str0_list = list(self.umls.cui2str[cui0])
        str1_list = list(self.umls.cui2str[cui1])
        if len(str0_list) > self.max_lui_per_cui:
            str0_list = sample(str0_list, self.max_lui_per_cui)
        if len(str1_list) > self.max_lui_per_cui:
            str1_list = sample(str1_list, self.max_lui_per_cui)
        use_len = min(len(str0_list), len(str1_list))
        str0_list = str0_list[0:use_len]
        str1_list = str1_list[0:use_len]

        sty0_index = self.sty2id[self.umls.cui2sty[cui0]]
        sty1_index = self.sty2id[self.umls.cui2sty[cui1]]

        str2_list = []
        cui2_index_list = []
        sty2_index_list = []

        cui2 = my_sample(self.umls.cui, self.umls.cui_count,
                         index * self.max_lui_per_cui, use_len * 2)
        sample_index = 0
        while len(str2_list) < use_len:
            if sample_index < len(cui2):
                use_cui2 = cui2[sample_index]
            else:
                sample_index = 0
                cui2 = my_sample(self.umls.cui, self.umls.cui_count,
                                 index * self.max_lui_per_cui, use_len * 2)
                use_cui2 = cui2[sample_index]
            # if not "\t".join([cui0, use_cui2, re, rel]) in self.umls.rel: # TOO SLOW!
            if True:
                cui2_index_list.append(self.cui2id[use_cui2])
                sty2_index_list.append(
                    self.sty2id[self.umls.cui2sty[use_cui2]])
                str2_list.append(sample(self.umls.cui2str[use_cui2], 1)[0])
                sample_index += 1

        # print(str0_list)
        # print(str1_list)
        # print(str2_list)

        input_ids = [self.tokenize_one(s)
                     for s in str0_list + str1_list + str2_list]
        input_ids = pad(input_ids, self.max_length)
        input_ids_0 = input_ids[0:use_len]
        input_ids_1 = input_ids[use_len:2 * use_len]
        input_ids_2 = input_ids[2 * use_len:]

        cui0_index = self.cui2id[cui0]
        cui1_index = self.cui2id[cui1]

        re_index = self.re2id[re]
        rel_index = self.rel2id[rel]
        return input_ids_0, input_ids_1, input_ids_2, \
            [cui0_index] * use_len, [cui1_index] * use_len, cui2_index_list, \
            [sty0_index] * use_len, [sty1_index] * use_len, sty2_index_list, \
            [re_index] * use_len, \
            [rel_index] * use_len

    def __len__(self):
        return self.len


def fixed_length_dataloader(umls_dataset, fixed_length=96, num_workers=0):
    base_sampler = RandomSampler(umls_dataset)
    batch_sampler = FixedLengthBatchSampler(
        sampler=base_sampler, fixed_length=fixed_length, drop_last=True)
    dataloader = DataLoader(umls_dataset, batch_sampler=batch_sampler,
                            collate_fn=my_collate_fn, num_workers=num_workers, pin_memory=True)
    return dataloader


if __name__ == "__main__":
    umls_dataset = UMLSDataset(umls_folder="../umls",
                               model_name_or_path="../biobert_v1.1",
                               lang=None)
    ipdb.set_trace()
    umls_dataloader = fixed_length_dataloader(umls_dataset, num_workers=4)
    now_time = time()
    for index, batch in enumerate(umls_dataloader):
        print(time() - now_time)
        now_time = time()
        if index < 10:
            for item in batch:
                print(item.shape)
            #print(batch)
        else:
            import sys
            sys.exit()
