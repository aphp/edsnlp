import warnings
from pathlib import Path
from typing import Dict, List, Union

import context  # noqa
import mlconjug3
import pandas as pd
import typer

from edsnlp.pipes.qualifiers.hypothesis.patterns import verbs_eds, verbs_hyp
from edsnlp.pipes.qualifiers.negation.patterns import verbs as neg_verbs
from edsnlp.pipes.qualifiers.reported_speech.patterns import verbs as rspeech_verbs

warnings.filterwarnings("ignore")


def conjugate_verb(
    verb: str,
    conjugator: mlconjug3.Conjugator,
) -> pd.DataFrame:
    """
    Conjugates the verb using an instance of mlconjug3,
    and formats the results in a pandas `DataFrame`.

    Parameters
    ----------
    verb : str
        Verb to conjugate.
    conjugator : mlconjug3.Conjugator
        mlconjug3 instance for conjugating.

    Returns
    -------
    pd.DataFrame
        Normalized dataframe containing all conjugated forms
        for the verb.
    """

    df = pd.DataFrame(
        conjugator.conjugate(verb).iterate(),
        columns=["mode", "tense", "person", "term"],
    )

    df.term = df.term.fillna(df.person)
    df.loc[df.person == df.term, "person"] = None

    df.insert(0, "verb", verb)

    return df


def conjugate(
    verbs: Union[str, List[str]],
    language: str = "fr",
) -> pd.DataFrame:
    """
    Conjugate a list of verbs.

    Parameters
    ----------
    verbs : Union[str, List[str]]
        List of verbs to conjugate
    language: str
        Language to conjugate. Defaults to French (`fr`).

    Returns
    -------
    pd.DataFrame
        Dataframe containing the conjugations for the provided verbs.
        Columns: `verb`, `mode`, `tense`, `person`, `term`
    """
    if isinstance(verbs, str):
        verbs = [verbs]

    conjugator = mlconjug3.Conjugator(language=language)

    df = pd.concat([conjugate_verb(verb, conjugator=conjugator) for verb in verbs])

    df = df.reset_index(drop=True)

    return df


def get_conjugated_verbs(
    verbs: Union[str, List[str]],
    matches: Union[List[Dict[str, str]], Dict[str, str]],
    language: str = "fr",
) -> List[str]:
    """
    Get a list of conjugated verbs.

    Parameters
    ----------
    verbs : Union[str, List[str]]
        List of verbs to conjugate.
    matches : Union[List[Dict[str, str]], Dict[str, str]]
        List of dictionary describing the mode/tense/persons to keep.
    language : str, optional
        [description], by default "fr" (French)

    Returns
    -------
    List[str]
        List of terms to look for.

    Examples
    --------
    >>> get_conjugated_verbs(
            "aimer",
            dict(mode="Indicatif", tense="Présent", person="1p"),
        )
    ['aimons']
    """

    if isinstance(matches, dict):
        matches = [matches]

    terms = []

    df = conjugate(
        verbs=verbs,
        language=language,
    )

    for match in matches:
        q = " & ".join([f'{k} == "{v}"' for k, v in match.items()])
        terms.extend(df.query(q).term.unique())

    return list(set(terms))


def conjugate_verbs(
    output_path: Path = typer.Argument(
        "edsnlp/resources/verbs.csv.gz", help="Path to the output CSV table."
    ),
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
