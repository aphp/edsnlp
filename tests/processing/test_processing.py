from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import pytest
from spacy.tokens import Doc

text = """
Motif :
Le patient est admis le 29 août 2020 pour des difficultés respiratoires.

Antécédents familiaux :
Le père est asthmatique, sans traitement particulier.

HISTOIRE DE LA MALADIE
Le patient dit avoir de la toux. \
Elle a empiré jusqu'à nécessiter un passage aux urgences.
La patiente avait un SOFA à l'admission de 8.

Conclusion
Possible infection au coronavirus
"""


def note(module):
    from pyspark.sql import types as T
    from pyspark.sql.session import SparkSession

    data = [(i, i // 5, text, datetime(2021, 1, 1)) for i in range(20)]

    if module == "pandas":
        return pd.DataFrame(
            data=data, columns=["note_id", "person_id", "note_text", "note_datetime"]
        )

    note_schema = T.StructType(
        [
            T.StructField("note_id", T.IntegerType()),
            T.StructField("person_id", T.IntegerType()),
            T.StructField("note_text", T.StringType()),
            T.StructField(
                "note_datetime",
                T.TimestampType(),
            ),
        ]
    )

    spark = SparkSession.builder.getOrCreate()
    notes = spark.createDataFrame(data=data, schema=note_schema)
    if module == "pyspark":
        return notes

    if module == "koalas":
        return notes.to_koalas()


@pytest.fixture
def model(blank_nlp):
    # Creates the spaCy instance
    nlp = blank_nlp

    # Normalisation of accents, case and other special characters
    nlp.add_pipe("eds.normalizer")

    # Extraction of named entities
    nlp.add_pipe(
        "eds.matcher",
        config=dict(
            terms=dict(
                respiratoire=[
                    "difficultes respiratoires",
                    "asthmatique",
                    "toux",
                ]
            ),
            regex=dict(
                covid=r"(?i)(?:infection\sau\s)?(covid[\s\-]?19|corona[\s\-]?virus)",
                traitement=r"(?i)traitements?|medicaments?",
                respiratoire="respiratoires",
            ),
            attr="NORM",
        ),
    )

    # Qualification of matched entities
    nlp.add_pipe("eds.negation")
    nlp.add_pipe("eds.hypothesis")
    nlp.add_pipe("eds.family")
    nlp.add_pipe("eds.reported_speech")
    nlp.add_pipe("eds.sofa")
    nlp.add_pipe("eds.dates")

    return nlp


params = [
    dict(module="pandas", n_jobs=1),
    dict(module="pandas", n_jobs=-2),
    dict(module="pyspark", n_jobs=None),
]

try:
    import databricks.koalas  # noqa F401

    params.append(dict(module="koalas", n_jobs=None))
except ImportError:
    pass


@pytest.mark.parametrize("param", params)
def test_pipelines(param, model):
    from pyspark.sql import types as T

    from edsnlp.processing import pipe

    module = param["module"]

    note_nlp = pipe(
        note(module=module),
        nlp=model,
        n_jobs=param["n_jobs"],
        context=["note_datetime"],
        extensions={
            "score_method": T.StringType(),
            "negation": T.BooleanType(),
            "hypothesis": T.BooleanType(),
            "family": T.BooleanType(),
            "reported_speech": T.BooleanType(),
            "date.year": T.IntegerType(),
            "date.month": T.IntegerType(),
        }
        if module in ("pyspark", "koalas")
        else [
            "score_method",
            "negation",
            "hypothesis",
            "family",
            "reported_speech",
            "date_year",
            "date_month",
        ],
        additional_spans=["dates"],
    )

    if module == "pyspark":
        note_nlp = note_nlp.toPandas()
    elif module == "koalas":
        note_nlp = note_nlp.to_pandas()

    assert len(note_nlp) == 140
    assert set(note_nlp.columns) == set(
        (
            "note_id",
            "lexical_variant",
            "label",
            "span_type",
            "start",
            "end",
            "negation",
            "hypothesis",
            "reported_speech",
            "family",
            "score_method",
            "date_year",
            "date_month",
        )
    )


def test_spark_missing_types(model):
    from edsnlp.processing import pipe

    with pytest.warns(Warning) as warned:
        pipe(
            note(module="pyspark"),
            nlp=model,
            extensions={"negation", "hypothesis", "family"},
        )
    assert any(
        "The following schema was inferred" in str(warning.message)
        for warning in warned
    )


@pytest.mark.parametrize("param", params)
def test_arbitrary_callback(param, model):
    from pyspark.sql import types as T

    from edsnlp.processing import pipe

    # We need to test PySpark with an installed function
    def dummy_extractor(doc: Doc) -> List[Dict[str, Any]]:
        return [
            dict(
                snippet=ent.text,
                length=len(ent.text),
                note_datetime=doc._.note_datetime,
            )
            for ent in doc.ents
        ]

    module = param["module"]

    note_nlp = pipe(
        note(module=module),
        nlp=model,
        n_jobs=param["n_jobs"],
        context=["note_datetime"],
        results_extractor=dummy_extractor,
        dtypes={
            "snippet": T.StringType(),
            "length": T.IntegerType(),
        },
    )

    if module == "pandas":
        assert set(note_nlp.columns) == {"snippet", "length", "note_datetime"}
        assert (note_nlp.snippet.str.len() == note_nlp.length).all()

    else:
        if module == "pyspark":
            note_nlp = note_nlp.toPandas()
        elif module == "koalas":
            note_nlp = note_nlp.to_pandas()

        assert set(note_nlp.columns) == {
            "note_id",
            "snippet",
            "length",
        }
        assert (note_nlp.snippet.str.len() == note_nlp.length).all()
