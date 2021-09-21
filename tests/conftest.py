import spacy
from pytest import fixture

import context
import edsnlp.components


@fixture(scope="session")
def nlp():
    model = spacy.blank("fr")

    model.add_pipe("sentences")
    model.add_pipe("pollution")
    model.add_pipe("sections")
    model.add_pipe(
        "matcher",
        config=dict(
            terms=dict(patient="patient"),
        ),
    )
    model.add_pipe(
        "matcher",
        name="matcher2",
        config=dict(
            terms=dict(anomalie="anomalie"),
        ),
    )
    model.add_pipe("hypothesis", config=dict(on_ents_only=False))
    model.add_pipe("negation", config=dict(on_ents_only=False))
    model.add_pipe("family", config=dict(on_ents_only=False))
    model.add_pipe("antecedents", config=dict(on_ents_only=False, use_sections=False))
    model.add_pipe("rspeech", config=dict(on_ents_only=False))

    return model


@fixture(scope="session")
def blank_nlp():
    model = spacy.blank("fr")
    model.add_pipe("sentences")
    return model


text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit.\n"
    "Pas d'anomalie détectée."
)


@fixture
def doc(nlp):
    return nlp(text)
