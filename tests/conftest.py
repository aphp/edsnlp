from datetime import datetime

import pandas as pd
import pytest
import spacy
from pytest import fixture

import edsnlp


@fixture(scope="session", params=["eds", "fr"])
def lang(request):
    return request.param


def make_nlp(lang):
    if lang == "eds":
        model = spacy.blank("eds")
    else:
        model = edsnlp.blank("fr")

    model.add_pipe("eds.normalizer")

    model.add_pipe("eds.sentences")
    model.add_pipe("eds.sections")

    model.add_pipe(
        "eds.matcher",
        config=dict(
            terms=dict(patient="patient"),
            attr="NORM",
            ignore_excluded=True,
        ),
    )
    model.add_pipe(
        "eds.matcher",
        name="matcher2",
        config=dict(
            regex=dict(anomalie=r"anomalie"),
        ),
    )

    model.add_pipe("eds.hypothesis")
    model.add_pipe("eds.negation")
    model.add_pipe("eds.family")
    model.add_pipe("eds.history")
    model.add_pipe("eds.reported_speech")

    model.add_pipe("eds.dates")
    model.add_pipe("eds.measurements")

    return model


@fixture(scope="session")
def nlp(lang):
    return make_nlp(lang)


@fixture(scope="session")
def nlp_eds():
    return make_nlp("eds")


@fixture
def blank_nlp(lang):
    if lang == "eds":
        model = spacy.blank("eds")
    else:
        model = edsnlp.blank("fr")
    model.add_pipe("eds.sentences")
    return model


def make_ml_pipeline():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.sentences", name="sentences")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
        ),
    )
    nlp.add_pipe(
        "eds.ner_crf",
        name="ner",
        config=dict(
            embedding=nlp.get_pipe("transformer"),
            mode="independent",
            target_span_getter=["ents", "ner-preds"],
            span_setter="ents",
        ),
    )
    ner = nlp.get_pipe("ner")
    ner.update_labels(["PERSON", "GIFT"])
    return nlp


@fixture()
def ml_nlp():
    return make_ml_pipeline()


@fixture(scope="session")
def frozen_ml_nlp():
    return make_ml_pipeline()


@fixture()
def text():
    return (
        "Le patient est admis pour des douleurs dans le bras droit, "
        "mais n'a pas de problème de locomotion. "
        "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
        "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNb"
        "NbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
        "Pourrait être un cas de rhume.\n"
        "Motif :\n"
        "Douleurs dans le bras droit.\n"
        "ANTÉCÉDENTS\n"
        "Le patient est déjà venu pendant les vacances\n"
        "d'été.\n"
        "Pas d'anomalie détectée."
    )


@fixture
def doc(nlp, text):
    return nlp(text)


@fixture
def blank_doc(blank_nlp, text):
    return blank_nlp(text)


@fixture
def df_notes():
    N_LINES = 100
    notes = pd.DataFrame(
        data={
            "note_id": list(range(N_LINES)),
            "note_text": N_LINES * [text],
            "note_datetime": N_LINES * [datetime.today()],
        }
    )

    return notes


def make_df_note(text, module):
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
        try:
            import databricks.koalas  # noqa F401
        except ImportError:
            pytest.skip("Koalas not installed")
        return notes.to_koalas()


@fixture
def df_notes_pandas(text):
    return make_df_note(text, "pandas")


@fixture
def df_notes_pyspark(text):
    return make_df_note(text, "pyspark")


@fixture
def df_notes_koalas(text):
    return make_df_note(text, "koalas")


@fixture
def run_in_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)
