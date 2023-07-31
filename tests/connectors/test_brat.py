import filecmp
import re
from os import listdir
from os.path import join
from pathlib import Path
from random import choice, randint, random
from string import ascii_letters, ascii_lowercase

import pytest
from spacy.language import Language

from edsnlp.connectors.brat import BratConnector


def random_word():
    n = randint(1, 20)
    return "".join(
        [choice(ascii_letters)] + [choice(ascii_lowercase) for _ in range(n)]
    )


def random_text():
    n = randint(30, 60)
    return " ".join([random_word() for _ in range(n)])


def random_brat_file(text):
    brat = []

    for match in re.finditer(r"\w+", text):
        if random() > 0.8:
            line = (
                f"T{len(brat) + 1}\tTEST {match.start()} {match.end()}\t{match.group()}"
            )
            brat.append(line)

    return "\n".join(brat) + "\n"


@pytest.fixture
def brat_folder(tmpdir):
    for i in range(100):
        text = random_text()
        brat = random_brat_file(text)

        with open(join(tmpdir, f"{i}.txt"), "w") as f:
            f.write(text)

        with open(join(tmpdir, f"{i}.ann"), "w") as f:
            if i == 0:
                f.write("\n")
            else:
                f.write(brat)

    return tmpdir


@pytest.fixture
def brat1(brat_folder) -> BratConnector:
    return BratConnector(brat_folder)


@pytest.fixture
def brat2(tmpdir) -> BratConnector:
    return BratConnector(tmpdir)


@pytest.fixture
def brat_importer():
    brat_dir = Path(__file__).parent.parent.resolve() / "resources" / "brat_data"
    return BratConnector(str(brat_dir))


@pytest.fixture
def brat_exporter(tmpdir):
    return BratConnector(tmpdir, attributes=["etat", "assertion"])


def test_empty_brat(brat2: BratConnector, blank_nlp: Language):
    with pytest.raises(AssertionError):
        brat2.brat2docs(blank_nlp)


def test_brat2pandas(brat1: BratConnector):
    brat1.get_brat()


def test_brat2brat(brat1: BratConnector, brat2: BratConnector, blank_nlp: Language):
    docs = brat1.brat2docs(blank_nlp)
    brat2.docs2brat(docs)

    files = listdir(brat1.directory)

    assert files

    for file in files:
        assert file in listdir(brat2.directory)
        assert filecmp.cmp(join(brat1.directory, file), join(brat2.directory, file))


def test_docs2brat(nlp, brat2):
    text = (
        "Le patient est admis pour des douleurs dans le bras droit, "
        "mais n'a pas de problème de locomotion. "
        "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
        "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbN"
        "BWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
        "Pourrait être un cas de rhume.\n"
        "Motif :\n"
        "Douleurs dans le bras droit."
    )

    doc1 = nlp(text)
    doc1.ents = doc1.spans["pollutions"]

    doc2 = nlp(text)
    doc2.ents = doc2.spans["section_titles"]

    docs = [doc1, doc2]
    for i, doc in enumerate(docs):
        doc._.note_id = i + 1

    brat2.docs2brat(docs)


def test_brat(
    brat_importer: BratConnector, brat_exporter: BratConnector, blank_nlp: Language
):
    doc = brat_importer.brat2docs(blank_nlp)[0]
    assert doc._.note_id == "subfolder/doc-1"

    attrs = ("etat", "assertion")
    spans_and_attributes = {
        "__ents__": sorted(
            [
                (e.start, e.end, e.text, tuple(getattr(e._, key) for key in attrs))
                for e in doc.ents
            ]
        ),
        **{
            name: sorted(
                [
                    (e.start, e.end, e.text, tuple(getattr(e._, key) for key in attrs))
                    for e in doc.spans[name]
                ]
            )
            for name in doc.spans
        },
    }

    assert spans_and_attributes == {
        "__ents__": [
            (6, 7, "douleurs", (None, None)),
            (7, 11, "dans le bras droit", (None, None)),
            (17, 21, "problème \nde locomotion", (None, "absent")),
            (25, 26, "AVC", ("passé", "non-associé")),
            (35, 36, "rhume", ("présent", "hypothétique")),
            (45, 46, "rhume", ("présent", "hypothétique")),
            (51, 52, "Douleurs", (None, None)),
            (52, 56, "dans le bras droit", (None, None)),
            (68, 69, "anomalie", (None, "absent")),
        ],
        "anatomie": [
            (9, 11, "bras droit", (None, None)),
            (54, 56, "bras droit", (None, None)),
        ],
        "localisation": [
            (7, 11, "dans le bras droit", (None, None)),
            (52, 56, "dans le bras droit", (None, None)),
        ],
        "pathologie": [
            (17, 21, "problème \nde locomotion", (None, "absent")),
            (25, 26, "AVC", ("passé", "non-associé")),
            (35, 36, "rhume", ("présent", "hypothétique")),
            (45, 46, "rhume", ("présent", "hypothétique")),
        ],
        "sosy": [
            (6, 7, "douleurs", (None, None)),
            (51, 52, "Douleurs", (None, None)),
            (68, 69, "anomalie", (None, "absent")),
        ],
    }

    doc.ents[0]._.etat = "test"

    brat_exporter.docs2brat([doc])
    with open(brat_exporter.directory / "subfolder" / "doc-1.ann") as f:
        exported_ann_text = f.read()
    assert (
        exported_ann_text
        == """\
T1	sosy 30 38	douleurs
A1	etat T1 test
T2	localisation 39 57	dans le bras droit
T3	anatomie 47 57	bras droit
T4	pathologie 75 83;85 98	problème de locomotion
A2	assertion T4 absent
T5	pathologie 114 117	AVC
A3	etat T5 passé
A4	assertion T5 non-associé
T6	pathologie 159 164	rhume
A5	etat T6 présent
A6	assertion T6 hypothétique
T7	pathologie 291 296	rhume
A7	etat T7 présent
A8	assertion T7 hypothétique
T8	sosy 306 314	Douleurs
T9	localisation 315 333	dans le bras droit
T10	anatomie 323 333	bras droit
T11	sosy 378 386	anomalie
A9	assertion T11 absent
"""
    )
