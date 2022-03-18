from pathlib import Path

import typer

import requests
import pandas as pd


def download_cities(
    output_path: Path = typer.Argument(
        "edsnlp/resources/cities.csv.gz",
        help="Path to the output CSV table.",
    ),
) -> None:
    """
    Convenience script to automatically download a list of French cities.
    """

    typer.echo("Downloading data...")

    r = requests.get(
        "https://www.data.gouv.fr/fr/datasets/r/34d4364c-22eb-4ac0-b179-7a1845ac033a"
    )

    df = pd.DataFrame.from_records(r.json())
    df = df[["codePostal", "nomCommune"]]
    df.columns = ["zip", "name"]

    typer.echo(f"Saving to {output_path}")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, index=False)

    typer.echo("Done !")


if __name__ == "__main__":
    typer.run(download_cities)
