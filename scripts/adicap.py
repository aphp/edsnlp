"""
Process ADICAP codes
Thésaurus de la codification ADICAP - Index raisonné des lésions
source : https://smt.esante.gouv.fr/terminologie-adicap/

"""

import gzip
import json
from pathlib import Path

import pandas as pd
import typer


def parse_each_dict(df, dictionaryCode: str):
    d_spec = df.query(f"dictionaryCode=='{dictionaryCode}'")

    decode_d_spec = {}

    for code, label in d_spec[["code", "label"]].values:
        decode_d_spec[code] = label

    d_value = decode_d_spec.pop(dictionaryCode)

    return dict(label=d_value, codes=decode_d_spec)


def get_decode_dict(df, dict_keys=["D1", "D2", "D3", "D4", "D5"]):
    decode_dict = {}
    for key in dict_keys:

        decode_dict[key] = parse_each_dict(df, dictionaryCode=key)

    return decode_dict


def run(
    raw: Path = typer.Argument(..., help="Path to the raw file"),
    output: Path = typer.Option(
        "edsnlp/resources/adicap.json.gz", help="Path to the output CSV table."
    ),
) -> None:
    """
    Convenience script to automatically process the ADICAP codes
    into a processable file.
    """

    df = pd.read_excel(
        raw,
        sheet_name="rawdatas",
        header=0,
    )

    decode_dict = get_decode_dict(df)

    typer.echo(f"Saving to {output}")

    with gzip.open(output, "w") as f:
        f.write(json.dumps(decode_dict).encode("utf-8"))

    typer.echo("Done !")


if __name__ == "__main__":
    typer.run(run)
