from datetime import datetime

import pandas as pd
import spacy
from pytest import fixture


@fixture(scope="session", params=["eds", "fr"])
def lang(request):
    return request.param


@fixture(scope="session")
def nlp(lang):
    model = spacy.blank(lang)

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


@fixture
def blank_nlp(lang):
    model = spacy.blank(lang)
    model.add_pipe("eds.sentences")
    return model


text = (
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
def doc(nlp):
    return nlp(text)


@fixture
def blank_doc(blank_nlp):
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
