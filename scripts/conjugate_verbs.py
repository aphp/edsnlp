import warnings
from pathlib import Path

import context  # noqa
import typer

from edsnlp.conjugator import conjugate
from edsnlp.pipelines.qualifiers.hypothesis.patterns import verbs_eds, verbs_hyp
from edsnlp.pipelines.qualifiers.negation.patterns import verbs as neg_verbs
from edsnlp.pipelines.qualifiers.reported_speech.patterns import verbs as rspeech_verbs

warnings.filterwarnings("ignore")


def conjugate_verbs(
    output_path: Path = typer.Argument(
        "edsnlp/resources/verbs.csv.gz", help="Path to the output CSV table."
    )
) -> None:
    """
    Convenience script to automatically conjugate a set of verbs,
    using mlconjug3 library.
    """

    all_verbs = set(neg_verbs + rspeech_verbs + verbs_eds + verbs_hyp)

    typer.echo(f"Conjugating {len(all_verbs)} verbs...")

    df = conjugate(list(all_verbs))

    typer.echo(f"Saving to {output_path}")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, index=False)

    typer.echo("Done !")


if __name__ == "__main__":
    typer.run(conjugate_verbs)
