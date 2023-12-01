import os
import typer
os.environ["OMP_NUM_THREADS"] = "16"
from extract_measurements_from_brat import ExtractMeasurements
from config import *
from pathlib import Path

import pandas as pd


def extract_measurements_cli(
    input_dir: Path,
    output_dir: Path,
):
    df = ExtractMeasurements(
        regex_convert_spans = measurements_pipe_regex_convert_spans,
        label_key = measurements_pipe_label_key,
        labels_to_remove = measurements_pipe_labels_to_remove,
        labels_linkable_to_measurement = measurements_pipe_labels_linkable_to_measurement,
        config_normalizer_from_label_key = measurements_pipe_config_normalizer_from_label_key,
        config_measurements_from_label_key = measurements_pipe_config_measurements_from_label_key,
        config_normalizer_from_tables = measurements_pipe_config_normalizer_from_tables,
        config_measurements_from_tables = measurements_pipe_config_measurements_from_tables,
    )(brat_dir = input_dir, only_tables = measurements_only_tables)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_json(output_dir / "pred_with_extraction.json")


if __name__ == "__main__":
    typer.run(extract_measurements_cli)
