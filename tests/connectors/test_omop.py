import random
import re
from random import choice, randint
from random import random as randomf
from string import ascii_letters, ascii_lowercase

import pandas as pd
import pytest

from edsnlp.connectors.omop import OmopConnector

random.seed(10)


def random_word():
    n = randint(1, 20)
    return "".join(
        [choice(ascii_letters)] + [choice(ascii_lowercase) for _ in range(n)]
    )


def random_text():
    n = randint(30, 60)
    return " ".join([random_word() for _ in range(n)])


def random_note_nlp(text):

    ents = []

    for match in re.finditer(r"\w+", text):
        if randomf() > 0.8:
            ent = dict(
                start_char=match.start(),
                end_char=match.end(),
                lexical_variant=match.group(),
                note_nlp_source_value=random_word().lower(),
                negation=randomf() > 0.5,
            )
            ents.append(ent)

    return ents


@pytest.fixture
def note():

    df = pd.DataFrame(dict(note_text=[random_text() for _ in range(10)]))
    df["note_id"] = range(len(df))
    df["note_datetime"] = "2021-10-19"

    return df


@pytest.fixture
def note_nlp(note):

    df = note.copy()
    df["ents"] = df.note_text.apply(random_note_nlp)
    df = df.explode("ents")
    df = pd.concat([df, df.ents.apply(pd.Series)], axis=1)
    df = df.drop(columns=["note_text", "ents", "note_datetime"])
    df["note_nlp_id"] = range(len(df))

    return df


@pytest.fixture
def omop(blank_nlp) -> OmopConnector:
    blank_nlp.add_pipe("negation")
    return OmopConnector(blank_nlp)


@pytest.fixture
def docs(omop: OmopConnector, note, note_nlp):
    return omop.omop2docs(note, note_nlp, extensions=["negation"])


def test_omop2docs(docs, note, note_nlp):

    lexical_variants = note_nlp.groupby("note_id")["lexical_variant"].agg(list)

    for doc, text, lvs in zip(docs, note.note_text, lexical_variants):
        assert doc.text == text
        assert len(doc.ents) == len(lvs)

        for ent, lv in zip(doc.ents, lvs):
            assert ent.text == lv


def test_docs2omop(omop: OmopConnector, docs):
    note, note_nlp = omop.docs2omop(docs, extensions=["negation"])

    lexical_variants = note_nlp.groupby("note_id")["lexical_variant"].agg(list)

    for doc, text, lvs in zip(docs, note.note_text, lexical_variants):
        assert doc.text == text
        assert len(doc.ents) == len(lvs)

        for ent, lv in zip(doc.ents, lvs):
            assert ent.text == lv


def test_roundtrip(omop: OmopConnector, docs, note, note_nlp):
    note2, note_nlp2 = omop.docs2omop(docs, extensions=["negation"])

    assert (note2 == note[note2.columns]).all().all()
    assert (note_nlp2 == note_nlp[note_nlp2.columns]).all().all()
