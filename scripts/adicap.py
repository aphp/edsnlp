"""
Process ADICAP codes
Thésaurus de la codification ADICAP - Index raisonné des lésions
source : https://smt.esante.gouv.fr/terminologie-adicap/

"""

import gzip
import json
import re
from pathlib import Path

import pandas as pd
import typer


def parse_each_dict(df, dictionaryCode: str):
    d_spec = df.query(f"dictionaryCode=='{dictionaryCode}'")
    d_spec.fillna("", inplace=True)

    decode_d_spec = {}

    for code, label, anatomyCode in d_spec[["code", "label", "anatomyCode"]].values:
        if dictionaryCode == "D5":
            if re.match(r"[0-9]{4}", code) is None:
                decode_d_spec[str(anatomyCode) + str(code)] = label
        else:
            decode_d_spec[str(anatomyCode) + str(code)] = label

    d_value = decode_d_spec.pop(dictionaryCode)

    return dict(label=d_value, codes=decode_d_spec)


def get_decode_dict(df, dict_keys=["D1", "D2", "D3", "D4", "D5", "D6", "D7"]):
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
