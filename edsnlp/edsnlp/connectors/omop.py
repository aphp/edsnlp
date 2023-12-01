from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.utils.filter import filter_spans


def omop2docs(
    note: pd.DataFrame,
    note_nlp: pd.DataFrame,
    nlp: Language,
    extensions: Optional[List[str]] = None,
) -> List[Doc]:
    """
    Transforms an OMOP-formatted pair of dataframes into a list of documents.

    Parameters
    ----------
    note : pd.DataFrame
        The OMOP `note` table.
    note_nlp : pd.DataFrame
        The OMOP `note_nlp` table
    nlp : Language
        spaCy language object.
    extensions : Optional[List[str]], optional
        Extensions to keep, by default None

    Returns
    -------
    List[Doc] :
        List of spaCy documents
    """

    note = note.copy()
    note_nlp = note_nlp.copy()

    extensions = extensions or []

    def row2ent(row):
        d = dict(
            start_char=row.start_char,
            end_char=row.end_char,
            label=row.get("note_nlp_source_value"),
            extensions={ext: row.get(ext) for ext in extensions},
        )

        return d

    # Create entities
    note_nlp["ents"] = note_nlp.apply(row2ent, axis=1)

    note_nlp = note_nlp.groupby("note_id", as_index=False)["ents"].agg(list)

    note = note.merge(note_nlp, on=["note_id"], how="left")

    # Generate documents
    note["doc"] = note.note_text.apply(nlp)

    # Process documents
    for _, row in note.iterrows():

        doc = row.doc
        doc._.note_id = row.note_id
        doc._.note_datetime = row.get("note_datetime")

        ents = []

        if not isinstance(row.ents, list):
            continue

        for ent in row.ents:

            span = doc.char_span(
                ent["start_char"],
                ent["end_char"],
                ent["label"],
                alignment_mode="expand",
            )

            for k, v in ent["extensions"].items():
                setattr(span._, k, v)

            ents.append(span)

            if span.label_ not in doc.spans:
                doc.spans[span.label_] = [span]
            else:
                doc.spans[span.label_].append(span)

        ents, discarded = filter_spans(ents, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

    return list(note.doc)


def docs2omop(
    docs: List[Doc],
    extensions: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms a list of spaCy docs to a pair of OMOP tables.

    Parameters
    ----------
    docs : List[Doc]
        List of documents to transform.
    extensions : Optional[List[str]], optional
        Extensions to keep, by default None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Pair of OMOP tables (`note` and `note_nlp`)
    """

    df = pd.DataFrame(dict(doc=docs))

    df["note_text"] = df.doc.apply(lambda doc: doc.text)
    df["note_id"] = df.doc.apply(lambda doc: doc._.note_id)
    df["note_datetime"] = df.doc.apply(lambda doc: doc._.note_datetime)

    if df.note_id.isna().any():
        df["note_id"] = range(len(df))

    df["ents"] = df.doc.apply(lambda doc: list(doc.ents))
    df["ents"] += df.doc.apply(lambda doc: list(doc.spans["discarded"]))

    note = df[["note_id", "note_text", "note_datetime"]]

    df = df[["note_id", "ents"]].explode("ents")

    extensions = extensions or []

    def ent2dict(
        ent: Span,
    ) -> Dict[str, Any]:

        d = dict(
            start_char=ent.start_char,
            end_char=ent.end_char,
            note_nlp_source_value=ent.label_,
            lexical_variant=ent.text,
            # normalized_variant=ent._.normalized.text,
        )

        for ext in extensions:
            d[ext] = getattr(ent._, ext)

        return d

    df["ents"] = df.ents.apply(ent2dict)

    columns = [
        "start_char",
        "end_char",
        "note_nlp_source_value",
        "lexical_variant",
        # "normalized_variant",
    ]
    columns += extensions

    df[columns] = df.ents.apply(pd.Series)

    df["term_modifiers"] = ""

    for i, ext in enumerate(extensions):
        if i > 0:
            df.term_modifiers += ";"
        df.term_modifiers += ext + "=" + df[ext].astype(str)

    df["note_nlp_id"] = range(len(df))

    note_nlp = df[["note_nlp_id", "note_id"] + columns]

    return note, note_nlp


class OmopConnector(object):
    """
    [summary]

    Parameters
    ----------
    nlp : Language
        spaCy language object.
    start_char : str, optional
        Name of the column containing the start character index of the entity,
        by default "start_char"
    end_char : str, optional
        Name of the column containing the end character index of the entity,
        by default "end_char"
    """

    def __init__(
        self,
        nlp: Language,
        start_char: str = "start_char",
        end_char: str = "end_char",
    ):

        self.start_char = start_char
        self.end_char = end_char

        self.nlp = nlp

    def preprocess(
        self, note: pd.DataFrame, note_nlp: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the input OMOP tables: modification of the column names.

        Parameters
        ----------
        note : pd.DataFrame
            OMOP `note` table.
        note_nlp : pd.DataFrame
            OMOP `note_nlp` table.

        Returns
        -------
        note : pd.DataFrame
            OMOP `note` table.
        note_nlp : pd.DataFrame
            OMOP `note_nlp` table.
        """

        note_nlp = note_nlp.rename(
            columns={
                self.start_char: "start_char",
                self.end_char: "end_char",
            }
        )

        return note, note_nlp

    def postprocess(
        self, note: pd.DataFrame, note_nlp: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Postprocess the input OMOP tables: modification of the column names.

        Parameters
        ----------
        note : pd.DataFrame
            OMOP `note` table.
        note_nlp : pd.DataFrame
            OMOP `note_nlp` table.

        Returns
        -------
        note : pd.DataFrame
            OMOP `note` table.
        note_nlp : pd.DataFrame
            OMOP `note_nlp` table.
        """

        note_nlp = note_nlp.rename(
            columns={
                "start_char": self.start_char,
                "end_char": self.end_char,
            }
        )

        return note, note_nlp

    def omop2docs(
        self,
        note: pd.DataFrame,
        note_nlp: pd.DataFrame,
        extensions: Optional[List[str]] = None,
    ) -> List[Doc]:
        """
        Transforms OMOP tables to a list of spaCy documents.

        Parameters
        ----------
        note : pd.DataFrame
            OMOP `note` table.
        note_nlp : pd.DataFrame
            OMOP `note_nlp` table.
        extensions : Optional[List[str]], optional
            Extensions to keep, by default None

        Returns
        -------
        List[Doc]
            List of spaCy documents.
        """
        note, note_nlp = self.preprocess(note, note_nlp)
        return omop2docs(note, note_nlp, self.nlp, extensions)

    def docs2omop(
        self,
        docs: List[Doc],
        extensions: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforms a list of spaCy documents to a pair of OMOP tables.

        Parameters
        ----------
        docs : List[Doc]
            List of spaCy documents.
        extensions : Optional[List[str]], optional
            Extensions to keep, by default None

        Returns
        -------
        note : pd.DataFrame
            OMOP `note` table.
        note_nlp : pd.DataFrame
            OMOP `note_nlp` table.
        """
        note, note_nlp = docs2omop(docs, extensions=extensions)
        note, note_nlp = self.postprocess(note, note_nlp)
        return note, note_nlp
