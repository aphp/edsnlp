import pandas as pd
from typing import List, Union, Tuple, Optional, Dict

from mlconjug3 import Conjugator

from pydantic import BaseModel


class Conjugation(BaseModel):
    mode: str
    tense: str
    person: Optional[str] = None
    term: str


def normalize(
    conjugation: Union[Tuple[str, str, str], Tuple[str, str, str, str]]
) -> Conjugation:
    """
    Normalize a conjugation tuple (infinitive and participles may display the person).

    Parameters
    ----------
    conjugation : Union[Tuple[str, str, str], Tuple[str, str, str, str]]
        Tuple of mode, tense, person (optional), variant.

    Returns
    -------
    Conjugation
        Normalized version, with the person always defined (possibly to `None`)
    """

    if len(conjugation) == 3:
        m, t, v = conjugation
        return Conjugation(
            mode=m,
            tense=t,
            term=v,
        )

    else:
        m, t, p, v = conjugation
        return Conjugation(
            mode=m,
            tense=t,
            person=p,
            term=v,
        )


def conjugate_verb(
    verb: str,
    language: str = "fr",
) -> pd.DataFrame:
    conjugator = Conjugator(language=language)

    conjugations = conjugator.conjugate(verb).iterate()

    df = pd.DataFrame.from_records(
        [normalize(conjugation).dict() for conjugation in conjugations]
    )

    df["verb"] = verb

    return df[["verb", "mode", "tense", "person", "term"]]


def conjugate(
    verbs: Union[List[str]],
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

    df = pd.concat([conjugate_verb(verb, language=language) for verb in verbs])

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
    >>> get_conjugated_verbs("aimer", dict(mode="Indicatif", tense="Présent", person="1p"))
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
        terms.extend(list(df.query(q).term))

    return list(set(terms))
