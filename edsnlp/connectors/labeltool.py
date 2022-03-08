from typing import List, Optional

import pandas as pd
from spacy.tokens import Doc
from tqdm import tqdm


def docs2labeltool(
    docs: List[Doc],
    extensions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Returns a labeltool-ready dataframe from a list of annotated document.

    Parameters
    ----------
    docs: list of spaCy Doc
        List of annotated spacy docs.
    extensions: list of extensions
        List of extensions to use by labeltool.

    Returns
    -------
    df: pd.DataFrame
        DataFrame tailored for labeltool.
    """

    if extensions is None:
        extensions = []

    entities = []

    for i, doc in enumerate(tqdm(docs, ascii=True, ncols=100)):
        for ent in doc.ents:
            d = dict(
                note_text=doc.text,
                offset_begin=ent.start_char,
                offset_end=ent.end_char,
                label_name=ent.label_,
                label_value=ent.text,
            )

            d["note_id"] = doc._.note_id or i

            for ext in extensions:
                d[ext] = getattr(ent._, ext)

            entities.append(d)

    df = pd.DataFrame.from_records(entities)

    columns = [
        "note_id",
        "note_text",
        "offset_begin",
        "offset_end",
        "label_name",
        "label_value",
    ]

    df = df[columns + extensions]

    return df
