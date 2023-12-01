"""
Process CIM10 patterns.

!!! warning "Watch out for the encoding"

    We had to convert the CIM-10 file from windows-1252 to utf-8.

Source: https://www.atih.sante.fr/plateformes-de-transmission-et-logiciels/logiciels-espace-de-telechargement/id_lot/456
"""  # noqa

from pathlib import Path

import pandas as pd
import typer


def run(
    raw: Path = typer.Argument(..., help="Path to the raw file"),
    output: Path = typer.Option(
        "edsnlp/resources/cim10.csv.gz", help="Path to the output CSV table."
    ),
) -> None:
    """
    Convenience script to automatically process the CIM10 terminology
    into a processable file.
    """

    df = pd.read_csv(raw, sep="|", header=None)

    typer.echo(f"Processing {len(df)} French ICD codes...")

    df.columns = ["code", "type", "ssr", "psy", "short", "long"]
    for column in ["code", "short", "long"]:
        df[column] = df[column].str.strip()

    typer.echo(f"Saving to {output}")

    df.to_csv(output, index=False)

    typer.echo("Done !")


if __name__ == "__main__":
    typer.run(run)
