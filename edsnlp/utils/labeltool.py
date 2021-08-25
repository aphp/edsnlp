from typing import List, Optional

import pandas as pd
from loguru import logger
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
    docs: list of Spacy Doc
        List of annotated spacy docs.
    extensions: list of extensions
        List of extensions to use by labeltool.

    Returns
    -------
    df: pd.DataFrame
        DataFrame tailored for labeltool.
    """

    note_id = Doc.has_extension("note_id")
    if not note_id:
        logger.info("note_id extension was not set.")

    if extensions is None:
        extensions = []

    entities = []

    for i, doc in enumerate(tqdm(docs, ascii=True, ncols=100)):
        for ent in doc.ents:
            d = dict(
                note_text=doc.text,
                offset_begin=ent.start_char,
                offset_end=ent.end_char,
                label=ent.label_,
                lexical_variant=ent.text,
            )

            if note_id:
                d["note_id"] = doc._.note_id or i
            else:
                d["note_id"] = i

            for ext in extensions:
                d[ext] = getattr(ent._, ext)

            entities.append(d)

    df = pd.DataFrame.from_records(entities)

    columns = [
        "note_id",
        "note_text",
        "offset_begin",
        "offset_end",
        "label",
        "lexical_variant",
    ]

    df = df[columns + extensions]

    return df
