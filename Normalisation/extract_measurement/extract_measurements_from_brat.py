import math
import random
import re
import sys
import time
from itertools import combinations
from os import listdir
from os.path import basename, isdir, isfile, join
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from extract_pandas_from_brat import extract_pandas
from measurements_patterns import (
    config_measurements_from_label_key,
    config_measurements_from_tables,
    config_normalizer_from_label_key,
    config_normalizer_from_tables,
    label_key,
    labels_linkable_to_measurement,
    labels_to_remove,
    regex_convert_spans,
)
from scipy.stats import bootstrap
from tqdm import tqdm
from typing_extensions import TypedDict

from edsnlp.processing import pipe


class UnitConfig(TypedDict):
    scale: float
    terms: List[str]
    followed_by: Optional[str] = None
    ui_decomposition: Dict[str, int]


class UnitlessRange(TypedDict):
    min: Optional[int]
    max: Optional[int]
    unit: str


class UnitlessPatternConfig(TypedDict):
    terms: List[str]
    ranges: List[UnitlessRange]


class SimpleMeasurementConfigWithoutRegistry(TypedDict):
    value_range: str
    value: Union[float, int]
    unit: str


class ValuelessPatternConfig(TypedDict):
    terms: Optional[List[str]]
    regex: Optional[List[str]]
    measurement: SimpleMeasurementConfigWithoutRegistry


class MeasureConfig(TypedDict):
    unit: str
    unitless_patterns: Optional[List[UnitlessPatternConfig]]
    valueless_patterns: Optional[List[ValuelessPatternConfig]]


class MeasurementsPipeConfig(TypedDict):
    measurements: Union[List[str], Tuple[str], Dict[str, MeasureConfig]]
    units_config: Dict[str, UnitConfig]
    number_terms: Dict[str, List[str]]
    value_range_terms: Dict[str, List[str]]
    all_measurements: bool
    parse_tables: bool
    parse_doc: bool
    stopwords_unitless: List[str]
    stopwords_measure_unit: List[str]
    measure_before_unit: bool
    unit_divisors: List[str]
    name: str
    ignore_excluded: bool
    attr: str


class ExtractMeasurements:
    def __init__(
        self,
        regex_convert_spans: Optional[str] = regex_convert_spans,
        label_key: Optional[str] = label_key,
        labels_to_remove: Optional[List[str]] = labels_to_remove,
        labels_linkable_to_measurement: Optional[
            List[str]
        ] = labels_linkable_to_measurement,
        config_normalizer_from_label_key: Optional[
            Dict[str, bool]
        ] = config_normalizer_from_label_key,
        config_measurements_from_label_key: Optional[
            MeasurementsPipeConfig
        ] = config_measurements_from_label_key,
        config_normalizer_from_tables: Optional[
            Dict[str, bool]
        ] = config_normalizer_from_tables,
        config_measurements_from_tables: Optional[
            MeasurementsPipeConfig
        ] = config_measurements_from_tables,
    ):
        print("--------------- Loading extraction pipe ---------------")
        self.regex_convert_spans = re.compile(regex_convert_spans)
        self.label_key = label_key
        self.labels_to_remove = labels_to_remove
        self.labels_linkable_to_measurement = labels_linkable_to_measurement
        self.nlp_from_label_key = self.load_nlp(
            config_normalizer_from_label_key, config_measurements_from_label_key
        )
        self.nlp_from_tables = self.load_nlp(
            config_normalizer_from_tables, config_measurements_from_tables
        )
        print("--------------- Extraction pipe loaded ---------------")

    def load_nlp(self, config_normalizer, config_measurements):
        nlp = spacy.blank("eds")
        nlp.add_pipe("eds.normalizer", config=config_normalizer)
        nlp.add_pipe("eds.dates")
        nlp.add_pipe("eds.tables")
        nlp.add_pipe("eds.measurements", config=config_measurements)
        return nlp

    def extract_pandas_labels_of_interest(self, brat_dir):
        # Convert span to list with span_start, span_end. It considers the new lines by adding one character.
        def convert_spans(span):
            span_match = self.regex_convert_spans.match(span)
            span_start = int(span_match.group(1))
            span_end = int(span_match.group(2))
            return [span_start, span_end]

        df = extract_pandas(IN_BRAT_DIR=brat_dir)
        df = df.loc[
            df["label"].isin(
                [self.label_key]
                + self.labels_to_remove
                + self.labels_linkable_to_measurement
            )
        ]
        df["span_converted"] = df["span"].apply(convert_spans)
        df = df[["term", "source", "span_converted", "label"]]
        return df

    @classmethod
    def is_overlapping(cls, a, b):
        # Return crop parts in a if a and b overlaps, 0, 0 if not
        if max(0, 1 + min(a[1], b[1]) - max(a[0], b[0])):
            return max(a[0], b[0]), min(a[1], b[1])
        else:
            return 0, 0

    def remove_labels_from_label_key(self, df):
        def get_parts_to_crop(old_parts_to_crop, new_part_to_crop):
            # From old_part_to_crops which is a list of segments and new_part_to_crop
            # which is a segment, return the union all the segments.
            # All segments in old_parts_to_crop are disjunct
            for old_part in old_parts_to_crop:
                crop_start, crop_end = self.is_overlapping(old_part, new_part_to_crop)
                if crop_start or crop_end:
                    return get_parts_to_crop(
                        old_parts_to_crop[1:],
                        [
                            min(old_part[0], new_part_to_crop[0]),
                            max(old_part[1], new_part_to_crop[1]),
                        ],
                    )
            old_parts_to_crop.append(new_part_to_crop)
            return old_parts_to_crop

        def crop_with_parts_to_crop(parts_to_crop, to_crop, to_crop_span):
            # parts_to_crop contains a list of segments to crop in to_crop (str)
            parts_to_crop.insert(0, [to_crop_span[0], to_crop_span[0]])
            parts_to_crop.append([to_crop_span[1], to_crop_span[1]])
            res = [
                to_crop[
                    parts_to_crop[i][1]
                    - to_crop_span[0] : parts_to_crop[i + 1][0]
                    - to_crop_span[0]
                ]
                for i in range(len(parts_to_crop) - 1)
            ]
            return "".join(res)

        label_key_df = df.loc[df["label"] == self.label_key].sort_values("source")
        specific_label_df = df.loc[
            df["label"].isin(
                self.labels_to_remove + self.labels_linkable_to_measurement
            )
        ]
        res = {"term_labels_removed": [], "terms_linked_to_measurement": []}
        label_keys = []
        source = None
        for label_key in tqdm(
            label_key_df.itertuples(index=False), total=label_key_df.shape[0]
        ):
            new_source = label_key.source
            if new_source != source:
                temp_df = specific_label_df.loc[
                    (specific_label_df["source"] == new_source)
                ]
                source = new_source
            labels_linkable_to_measurement = []
            parts_to_crop = []

            for label in temp_df.itertuples(index=False):

                crop_start, crop_end = self.is_overlapping(
                    label_key.span_converted, label.span_converted
                )

                if crop_start or crop_end:

                    if label.label in self.labels_to_remove:
                        if not parts_to_crop:
                            parts_to_crop.append([crop_start, crop_end])

                        else:
                            parts_to_crop = get_parts_to_crop(
                                parts_to_crop, [crop_start, crop_end]
                            )
                    if label.label in self.labels_linkable_to_measurement:
                        labels_linkable_to_measurement.append(label.term)
            res["term_labels_removed"].append(
                crop_with_parts_to_crop(
                    parts_to_crop, label_key.term, label_key.span_converted
                )
            )
            res["terms_linked_to_measurement"].append(labels_linkable_to_measurement)
            label_keys.append(label_key)
        res = pd.DataFrame(label_keys).join(pd.DataFrame(res))
        return res.reset_index(drop=True)

    def get_measurements_from_label_key(self, df):
        df_for_nlp_from_label_key = pd.DataFrame(
            {"note_text": df["term_labels_removed"], "note_id": df.index}
        )
        df_for_nlp_from_label_key = pipe(
            note=df_for_nlp_from_label_key,
            nlp=self.nlp_from_label_key,
            n_jobs=-1,
            additional_spans=["measurements"],
            extensions=["value"],
        )
        df_for_nlp_from_label_key = (
            df_for_nlp_from_label_key.groupby("note_id")
            .agg({"note_id": "first", "value": list, "lexical_variant": list})
            .reset_index(drop=True)
            .rename(columns={"value": "found"})
        )
        df = pd.merge(
            df, df_for_nlp_from_label_key, left_index=True, right_on="note_id"
        )
        df["found"] = df["found"].fillna("").apply(list)
        return df.reset_index(drop=True)

    def get_measurements_from_tables(
        self, df, df_labels_of_interest, brat_dir, only_tables
    ):

        # Treat each txt files
        txt_files = [
            f
            for f in listdir(brat_dir)
            if isfile(join(brat_dir, f))
            if f.endswith(".txt")
        ]
        ann_files = [f[:-3] + "ann" for f in txt_files]
        text_df = {"note_text": [], "note_id": []}
        for i, txt_file in enumerate(txt_files):
            with open(join(brat_dir, txt_file), "r") as file:
                text = file.read()
            text_df["note_text"].append(text)
            text_df["note_id"].append(txt_file[:-3] + "ann")
        text_df = pd.DataFrame(text_df)
        df_for_nlp_from_table = pipe(
            note=text_df,
            nlp=self.nlp_from_tables,
            n_jobs=-1,
            additional_spans=["measurements"],
            extensions=["value"],
        )
        # Load discriminative dataframe (in other words df containing terms with a label to remove) so that we can drop matches when one overlaps one of these words
        discriminative_df = df_labels_of_interest.loc[
            df_labels_of_interest["label"].isin(self.labels_to_remove)
        ]

        def get_measurements_from_tables_one_file(df_for_nlp_from_table, ann_file):
            # Select label_keys from the ann_file
            # and check if our matcher from tables
            # finds measurements with a span overlapping
            # one of these label_keys. If yes, then we keep this measurement
            # and throw all the ones found by our first matcher
            # from the label_key at stake.
            df_part = df.loc[df["source"] == ann_file].copy().reset_index(drop=True)
            df_part["new_found"] = [[] for _ in range(len(df_part))]
            discriminative_df_part = (
                discriminative_df.loc[discriminative_df["source"] == ann_file]
                .copy()
                .reset_index(drop=True)
            )
            df_for_nlp_from_table_part = (
                df_for_nlp_from_table.loc[df_for_nlp_from_table["note_id"] == ann_file]
                .copy()
                .reset_index(drop=True)
            )
            for measurement_from_table in df_for_nlp_from_table_part.itertuples(
                index=False
            ):
                measurement_span = [
                    measurement_from_table.start,
                    measurement_from_table.end,
                ]

                # Check if a match is in a term with a label to remove
                overlapping_discriminative_indexes = discriminative_df_part.loc[
                    discriminative_df_part["span_converted"].apply(
                        lambda x: self.is_overlapping(x, measurement_span)
                    )
                    != (0, 0)
                ].index.values.tolist()
                if overlapping_discriminative_indexes:
                    continue
                # Link the match measure to a label_key - label_linkable_to_measurement
                overlapping_label_key_indexes = df_part.loc[
                    df_part["span_converted"].apply(
                        lambda x: self.is_overlapping(x, measurement_span)
                    )
                    != (0, 0)
                ].index.values.tolist()
                for i in overlapping_label_key_indexes:
                    df_part.iloc[i]["new_found"].append(measurement_from_table.value)
            return df_part

        # DataFrame with merged doc and tables matches
        result_df_per_file = []
        for ann_file in tqdm(ann_files, total=len(ann_files)):
            result_df_per_file.append(
                get_measurements_from_tables_one_file(df_for_nlp_from_table, ann_file)
            )

        result_df = pd.concat(result_df_per_file, ignore_index=True)
        if only_tables:
            result_df = result_df.loc[result_df["new_found"].astype(bool)].reset_index(
                drop=True
            )
            result_df = result_df.drop(columns=["found"]).rename(
                columns={"new_found": "found"}
            )
            return result_df
        else:
            result_df["found"] = result_df.apply(
                lambda x: x["found"] * (not x["new_found"])
                + x["new_found"] * (len(x["new_found"]) > 0),
                axis=1,
            )
            return result_df.drop(columns=["new_found"])

    def prepare_df_for_normalization(self, df):
        # This method converts SimpleMeasurement objects to strings
        # So that It can be exported to json
        # Moreover, for each term, if any terms_linked_to_measurement are found,
        # We fill the cell with a list of 1 item:
        # this term cropped by the found measures from It

        # Fill empty terms_linked_to_measurement
        mask_empty_labels_linkable_to_measurement = (
            df["terms_linked_to_measurement"].str.len() == 0
        )
        df_empty_labels_linkable_to_measurement = df[
            mask_empty_labels_linkable_to_measurement
        ][["term", "lexical_variant"]]
        df.loc[
            mask_empty_labels_linkable_to_measurement, "terms_linked_to_measurement"
        ] = df_empty_labels_linkable_to_measurement.apply(
            lambda row: [
                re.compile(
                    r"\b(?:" + "|".join(row["lexical_variant"]) + r")\b", re.IGNORECASE
                ).sub("", row["term"])
            ],
            axis=1,
        )

        # Convert SimpleMeasurement to str
        df["found"] = df["found"].apply(
            lambda measurements: [
                measurement.value_range
                + " "
                + str(measurement.value)
                + " "
                + measurement.unit
                for measurement in measurements
            ]
        )
        df = df.drop(
            columns=["label", "term_labels_removed", "note_id", "lexical_variant"]
        )
        return df

    def __call__(self, brat_dir, only_tables):
        print(
            "--------------- Converting BRAT files to Pandas DataFrame... ---------------"
        )
        tic = time.time()
        df_labels_of_interest = self.extract_pandas_labels_of_interest(brat_dir)
        tac = time.time()
        print(f"Converting BRAT files to Pandas DataFrame : {tac-tic:.2f} sec")
        print("--------------- Removing labels from label keys... ---------------")
        tic = time.time()
        df = self.remove_labels_from_label_key(df_labels_of_interest)
        tac = time.time()
        print(f"Removing labels from label keys : {tac-tic:.2f} sec")
        print(
            "--------------- Extracting measurements from label keys... ---------------"
        )
        tic = time.time()
        df = self.get_measurements_from_label_key(df)
        tac = time.time()
        print(f"Extracting measurements from label keys : {tac-tic:.2f} sec")
        print("--------------- Extracting measurements from tables... ---------------")
        tic = time.time()
        df = self.get_measurements_from_tables(
            df, df_labels_of_interest, brat_dir, only_tables
        )
        tac = time.time()
        print(f"Extracting measurements from tables : {tac-tic:.2f} sec")
        print("--------------- Formatting table for normalization... ---------------")
        tic = time.time()
        df = self.prepare_df_for_normalization(df)
        tac = time.time()
        print(f"Formatting table for normalization : {tac-tic:.2f} sec")
        return df
