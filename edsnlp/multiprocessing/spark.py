from functools import partial
from typing import List, Optional, Tuple, Union

from pyspark.sql import functions as F
from pyspark.sql import types as T
from spacy import Language

from edsnlp.pipelines.base import BaseComponent


def udf_factory(
    nlp: Language,
    qualifiers: Optional[List[str]] = None,
    additional_spans: Union[List[str], str] = "discarded",
    additional_extensions: List[Tuple[str, T.DataType]] = [],
):

    if qualifiers is None:
        qualifiers = []

    schema = T.ArrayType(
        T.StructType(
            [
                T.StructField("lexical_variant", T.StringType(), False),
                T.StructField("label", T.StringType(), False),
                T.StructField("span_type", T.StringType(), True),
                T.StructField("start", T.IntegerType(), True),
                T.StructField("end", T.IntegerType(), False),
                *[
                    T.StructField(qualifier, T.BooleanType(), True)
                    for qualifier in qualifiers
                ],
                *[
                    T.StructField(*extension, True)
                    for extension in additional_extensions
                ],
            ]
        )
    )

    def f(
        text,
        additional_spans=additional_spans,
        additional_extensions=additional_extensions,
    ):

        if text is None:
            return []

        for _, pipe in nlp.pipeline:
            if isinstance(pipe, BaseComponent):
                pipe.set_extensions()

        doc = nlp(text)

        ents = []

        for ent in doc.ents:

            modifiers = [getattr(ent._, qualifier) for qualifier in qualifiers]
            extensions = [
                getattr(ent._, extension) for extension, _ in additional_extensions
            ]

            ents.append(
                (
                    ent.text,
                    ent.label_,
                    "ents",
                    ent.start_char,
                    ent.end_char,
                    *modifiers,
                    *extensions,
                )
            )

        if additional_spans is None:
            return ents

        if type(additional_spans) == str:
            additional_spans = [additional_spans]

        for spans_name in additional_spans:

            for ent in doc.spans.get(spans_name, []):

                modifiers = [getattr(ent._, qualifier) for qualifier in qualifiers]
                extensions = [
                    getattr(ent._, extension) for extension, _ in additional_extensions
                ]

                ents.append(
                    (
                        ent.text,
                        ent.label_,
                        spans_name,
                        ent.start_char,
                        ent.end_char,
                        *modifiers,
                        *extensions,
                    )
                )

        return ents

    f_udf = F.udf(
        partial(
            f,
            additional_spans=additional_spans,
            additional_extensions=additional_extensions,
        ),
        schema,
    )

    return f_udf


def apply_nlp(
    note,
    nlp,
    qualifiers=None,
    additional_spans: Union[List[str], str] = "discarded",
    additional_extensions: List[Tuple[str, T.DataType]] = [],
):

    matcher = udf_factory(
        nlp=nlp,
        qualifiers=qualifiers,
        additional_spans=additional_spans,
        additional_extensions=additional_extensions,
    )

    note_nlp = note.withColumn("matches", matcher(note.note_text))
    note_nlp = note_nlp.withColumn("matches", F.explode(note_nlp.matches))

    note_nlp = note_nlp.select("note_id", "matches.*")

    note_nlp = note_nlp.withColumn("note_nlp_datetime", F.current_timestamp())
    note_nlp = note_nlp.select(
        F.monotonically_increasing_id().alias("note_nlp_id"), "*"
    )

    return note_nlp
