#!/usr/bin/env python
# coding: utf-8
# %%

# # Build prediction file
#
# From our files with NER prediction, extract a pandas data frame to work on entities easily
#

# %%


import collections
import math
import re
from os import listdir
from os.path import basename, isdir, isfile, join

import numpy as np
import pandas as pd


def extract_pandas(IN_BRAT_DIR, OUT_DF=None, labels=None):

    assert isdir(IN_BRAT_DIR)

    ENTITY_REGEX = re.compile("^(.\d+)\t([^ ]+) ([^\t]+)\t(.*)$")

    data = []
    patients = []

    # extract all ann_files from IN_BRAT_DIR
    ann_files = [
        f
        for f in listdir(IN_BRAT_DIR)
        if isfile(join(IN_BRAT_DIR, f))
        if f.endswith(".ann")
    ]

    for ann_file in ann_files:
        ann_path = join(IN_BRAT_DIR, ann_file)
        txt_path = ann_path[:-4] + ".txt"

        # sanity check
        assert isfile(ann_path)
        assert isfile(txt_path)

        # Read text file to get patient number :
        with open(txt_path, "r", encoding="utf-8") as f_txt:
            lines_txt = f_txt.readlines()
        patient_num = lines_txt[0][:-1]
        patients.append(patient_num)

        # Read ann file
        with open(ann_path, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        for line in lines:
            entity_match = ENTITY_REGEX.match(line.strip())
            if entity_match is not None:
                ann_id = entity_match.group(1)
                label = entity_match.group(2)
                offsets = entity_match.group(3)
                term = entity_match.group(4)
                if labels is None:
                    data.append([ann_id, term, label, basename(ann_path), offsets])
                elif label in labels:
                    data.append([ann_id, term, label, basename(ann_path), offsets])

    columns = ["ann_id", "term", "label", "source", "span"]
    dataset_df = pd.DataFrame(data=list(data), columns=columns)
    if OUT_DF:
        dataset_df.to_csv(OUT_DF)

    return dataset_df
