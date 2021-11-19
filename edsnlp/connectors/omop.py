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

    # assert set(["note_id", "note_text"]) <= set(note.columns)
    # assert set(["note_id", "start_char", "end_char", "note_nlp_source_value"]) <= set(
    #     note.columns
    # )

    df = note.merge(note_nlp, on=["note_id"])

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
    df["ents"] = df.apply(row2ent, axis=1)

    df = df.groupby("note_id", as_index=False)["ents"].agg(list)

    df = df.merge(note, on=["note_id"])

    # Generate documents
    df["doc"] = df.note_text.apply(nlp)

    # Process documents
    for _, row in df.iterrows():

        doc = row.doc
        doc._.note_id = row.note_id
        doc._.note_datetime = row.get("note_datetime")

        ents = []

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

    return list(df.doc)


def docs2omop(
    docs: List[Doc],
    extensions: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.DataFrame(dict(doc=docs))

    df["note_text"] = df.doc.apply(lambda doc: doc.text)
    df["note_id"] = df.doc.apply(lambda doc: doc._.note_id)
    df["note_datetime"] = df.doc.apply(lambda doc: doc._.note_datetime)

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
    def __init__(
        self,
        nlp: Language,
        start_char: str = "start_char",
        end_char: str = "end_char",
    ) -> None:

        self.start_char = start_char
        self.end_char = end_char

        self.nlp = nlp

    def preprocess(self, note, note_nlp):

        note_nlp = note_nlp.rename(
            columns={
                self.start_char: "start_char",
                self.end_char: "end_char",
            }
        )

        return note, note_nlp

    def postprocess(self, note, note_nlp):

        note_nlp = note_nlp.rename(
            columns={
                "start_char": self.start_char,
                "end_char": self.end_char,
            }
        )

        return note, note_nlp

    def omop2docs(
        self,
        note,
        note_nlp,
        extensions: Optional[List[str]] = None,
    ) -> List[Doc]:
        note, note_nlp = self.preprocess(note, note_nlp)
        return omop2docs(note, note_nlp, self.nlp, extensions)

    def docs2omop(
        self,
        docs: List[Doc],
        extensions: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        note, note_nlp = docs2omop(docs, extensions=extensions)
        note, note_nlp = self.postprocess(note, note_nlp)
        return note, note_nlp
