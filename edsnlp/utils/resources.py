import gzip
import json
from functools import lru_cache
from typing import List, Optional

import pandas as pd

from edsnlp import BASE_DIR


def get_verbs(
    verbs: Optional[List[str]] = None, check_contains: bool = True
) -> pd.DataFrame:
    """
    Extract verbs from the resources, as a pandas dataframe.

    Parameters
    ----------
    verbs : List[str], optional
        List of verbs to keep. Returns all verbs by default.
    check_contains : bool, optional
        Whether to check that no verb is missing if a list of verbs was provided.
        By default True

    Returns
    -------
    pd.DataFrame
        DataFrame containing conjugated verbs.
    """

    conjugated_verbs = pd.read_csv(BASE_DIR / "resources" / "verbs.csv.gz")

    if not verbs:
        return conjugated_verbs

    verbs = set(verbs)

    selected_verbs = conjugated_verbs[conjugated_verbs.verb.isin(verbs)]

    if check_contains:
        assert len(verbs) == selected_verbs.verb.nunique(), "Some verbs are missing !"

    return selected_verbs


@lru_cache()
def get_adicap_dict():
    """
    Returns
    -------
    Dict
    """

    with gzip.open(BASE_DIR / "resources" / "adicap.json.gz", "r") as fin:
        decode_dict = json.loads(fin.read().decode("utf-8"))

    return decode_dict
