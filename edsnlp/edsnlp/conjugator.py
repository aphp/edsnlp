from typing import Dict, List, Union

import mlconjug3
import pandas as pd


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
