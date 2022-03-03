import pandas as pd
import pytest
import spacy
from pyspark.sql import types as T
from pyspark.sql.session import SparkSession

from edsnlp.processing import pipe

text = """
Motif :
Le patient est admis le 29 août pour des difficultés respiratoires.

Antécédents familiaux :
Le père est asthmatique, sans traitement particulier.

HISTOIRE DE LA MALADIE
Le patient dit avoir de la toux. \
Elle a empiré jusqu'à nécessiter un passage aux urgences.
La patiente avait un SOFA à l'admission de 8.

Conclusion
Possible infection au coronavirus
"""

spark = SparkSession.builder.getOrCreate()


def note(how: str):

    data = [(i, i // 5, text) for i in range(20)]

    if how == "spark":
        note_schema = T.StructType(
            [
                T.StructField("note_id", T.IntegerType()),
                T.StructField("person_id", T.IntegerType()),
                T.StructField("note_text", T.StringType()),
            ]
        )

        return spark.createDataFrame(data=data, schema=note_schema)

    else:
        return pd.DataFrame(data=data, columns=["note_id", "person_id", "note_text"])


@pytest.fixture
def model():
    # Creates the SpaCy instance
    nlp = spacy.blank("fr")

    # Normalisation of accents, case and other special characters
    nlp.add_pipe("normalizer")
    # Detecting end of lines
    nlp.add_pipe("sentences")

    # Extraction of named entities
    nlp.add_pipe(
        "matcher",
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
    nlp.add_pipe("negation")
    nlp.add_pipe("hypothesis")
    nlp.add_pipe("family")
    nlp.add_pipe("rspeech")
    nlp.add_pipe("SOFA")
    nlp.add_pipe("dates")

    return nlp


@pytest.mark.parametrize("how", ["simple", "parallel", "spark"])
def test_pipelines(how, model):

    note_nlp = pipe(
        note(how=how),
        nlp=model,
        how=how,
        extensions={
            "score_method": T.StringType(),
            "negation": T.BooleanType(),
            "hypothesis": T.BooleanType(),
            "family": T.BooleanType(),
            "reported_speech": T.BooleanType(),
            "parsed_date": T.TimestampType(),
        },
        additional_spans=["dates"],
    )

    if type(note_nlp) != pd.DataFrame:
        note_nlp = note_nlp.toPandas()

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
            "parsed_date",
        )
    )


def test_spark_missing_types(model):

    with pytest.raises(ValueError):
        pipe(
            note(how="spark"),
            nlp=model,
            how="spark",
            extensions={"negation", "hypothesis", "family"},
            additional_spans=["dates"],
        )
