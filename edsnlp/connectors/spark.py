from typing import List, Optional

from pyspark.sql import functions as F
from pyspark.sql import types as T
from spacy import Language

from edsnlp.base.component import BaseComponent


def udf_factory(
    nlp: Language,
    qualifiers: Optional[List[str]] = None,
    keep_discarded: bool = False,
):

    if qualifiers is None:
        qualifiers = []

    schema = T.ArrayType(
        T.StructType(
            [
                T.StructField("lexical_variant", T.StringType(), False),
                T.StructField("label", T.StringType(), False),
                T.StructField("discarded", T.BooleanType(), True),
                T.StructField("start", T.IntegerType(), True),
                T.StructField("end", T.IntegerType(), False),
                *[
                    T.StructField(qualifier, T.BooleanType(), True)
                    for qualifier in qualifiers
                ],
            ]
        )
    )

    @F.udf(schema)
    def f(text):

        if text is None:
            return []

        for _, pipe in nlp.pipeline:
            if isinstance(pipe, BaseComponent):
                pipe.set_extensions()

        doc = nlp(text)

        ents = []

        for ent in doc.ents:

            modifiers = [getattr(ent._, qualifier) for qualifier in qualifiers]

            ents.append(
                (
                    ent.text,
                    ent.label_,
                    False,
                    ent.start_char,
                    ent.end_char,
                    *modifiers,
                )
            )

        if keep_discarded:
            for ent in doc.spans.get("discarded", []):

                modifiers = [getattr(ent._, qualifier) for qualifier in qualifiers]

                ents.append(
                    (
                        ent.text,
                        ent.label_,
                        True,
                        ent.start_char,
                        ent.end_char,
                        *modifiers,
                    )
                )

        return ents

    return f


def apply_nlp(
    note,
    nlp,
    qualifiers=None,
    keep_discarded: bool = False,
):

    matcher = udf_factory(
        nlp=nlp,
        qualifiers=qualifiers,
        keep_discarded=keep_discarded,
    )

    note_nlp = note.withColumn("matches", matcher(note.note_text))
    note_nlp = note_nlp.withColumn("matches", F.explode(note_nlp.matches))

    note_nlp = note_nlp.select("note_id", "matches.*")

    note_nlp = note_nlp.withColumn("note_nlp_datetime", F.current_timestamp())
    note_nlp = note_nlp.select(
        F.monotonically_increasing_id().alias("note_nlp_id"), "*"
    )

    return note_nlp
