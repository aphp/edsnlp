import spacy
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.session import SparkSession
from pytest import fixture

from edsnlp.multiprocessing.spark import apply_nlp, udf_factory

text = """\
Motif :
Le patient est admis le 29 août pour des difficultés respiratoires.

Antécédents familiaux :
Le père est asthmatique, sans traitement particulier.

HISTOIRE DE LA MALADIE
Le patient dit avoir de la toux depuis trois jours. \
Elle a empiré jusqu'à nécessiter un passage aux urgences.
La patiente avait un SOFA à l'admission de 8.

Conclusion
Possible infection au coronavirus
"""

spark = SparkSession.builder.getOrCreate()


@fixture
def note():

    note_schema = T.StructType(
        [
            T.StructField("note_id", T.IntegerType()),
            T.StructField("person_id", T.IntegerType()),
            T.StructField("note_text", T.StringType()),
        ]
    )

    data = [(i, i // 5, text) for i in range(20)]

    return spark.createDataFrame(data=data, schema=note_schema)


@fixture
def model():
    # Creates the Spacy instance
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

    return nlp


def test_spark_udf(note, model):
    matcher = udf_factory(
        model,
        ["negated", "hypothesis", "reported_speech", "family"],
        additional_spans=["discarded"],
        additional_extensions=[("score_method", T.StringType())],
    )
    note_nlp = note.withColumn("matches", matcher(note.note_text))
    note_nlp = note_nlp.withColumn("matches", F.explode(note_nlp.matches))

    note_nlp = note_nlp.select("note_id", "person_id", "matches.*")

    df = note_nlp.toPandas()

    assert len(df) == 140
    assert set(df.columns) == set(
        (
            "note_id",
            "person_id",
            "lexical_variant",
            "label",
            "span_type",
            "start",
            "end",
            "negated",
            "hypothesis",
            "reported_speech",
            "family",
            "score_method",
        )
    )


def test_spark_pipeline(note, model):

    df = apply_nlp(
        note=note,
        nlp=model,
        qualifiers=["negated", "hypothesis", "reported_speech", "family"],
    ).toPandas()

    assert len(df) == 140
