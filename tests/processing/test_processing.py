from datetime import datetime

import databricks.koalas  # noqa F401
import pandas as pd
import pytest
import spacy
from pyspark.sql import types as T
from pyspark.sql.session import SparkSession

from edsnlp.processing import pipe
from edsnlp.processing.helpers import DataFrameModules

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


def note(module: DataFrameModules):

    data = [(i, i // 5, text, datetime(2021, 1, 1)) for i in range(20)]

    if module == DataFrameModules.PANDAS:
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
    if module == DataFrameModules.PYSPARK:
        return notes

    if module == DataFrameModules.KOALAS:
        return notes.to_koalas()


@pytest.fixture
def model(lang):
    # Creates the spaCy instance
    nlp = spacy.blank(lang)

    # Normalisation of accents, case and other special characters
    nlp.add_pipe("eds.normalizer")
    # Detecting end of lines
    nlp.add_pipe("eds.sentences")

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
    dict(module=DataFrameModules.PANDAS, n_jobs=1),
    dict(module=DataFrameModules.PANDAS, n_jobs=-2),
    dict(module=DataFrameModules.PYSPARK, n_jobs=None),
    dict(module=DataFrameModules.KOALAS, n_jobs=None),
]


@pytest.mark.parametrize("param", params)
def test_pipelines(param, model):

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
        },
        additional_spans=["dates"],
    )

    if module == DataFrameModules.PYSPARK:
        note_nlp = note_nlp.toPandas()
    elif module == DataFrameModules.KOALAS:
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

    with pytest.raises(ValueError):
        pipe(
            note(module=DataFrameModules.PYSPARK),
            nlp=model,
            extensions={"negation", "hypothesis", "family"},
        )


@pytest.mark.parametrize("param", params)
def test_arbitrary_callback(param, model):

    # We need to test PySpark with an installed function
    from edsnlp.processing.utils import dummy_extractor

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

    if module == DataFrameModules.PANDAS:
        assert set(note_nlp.columns) == {"snippet", "length", "note_datetime"}
        assert (note_nlp.snippet.str.len() == note_nlp.length).all()

    else:
        if module == DataFrameModules.PYSPARK:
            note_nlp = note_nlp.toPandas()
        elif module == DataFrameModules.KOALAS:
            note_nlp = note_nlp.to_pandas()

        assert set(note_nlp.columns) == {
            "note_id",
            "snippet",
            "length",
        }
        assert (note_nlp.snippet.str.len() == note_nlp.length).all()
