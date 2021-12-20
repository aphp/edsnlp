import filecmp
import re
from os import listdir
from os.path import join
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
