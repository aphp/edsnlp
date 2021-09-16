from edsnlp.utils.brat import BratConnector
import pytest

BRAT_FILES = "data/section_dataset"


@pytest.fixture(scope="module")
def brat():
    return BratConnector(BRAT_FILES)


@pytest.fixture
def brat2(tmpdir):
    return BratConnector(tmpdir)


# def test_brat2pandas(brat):
#     texts, annotations = brat.get_brat()


def test_docs2brat(nlp, brat2):
    text = (
        "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
        "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
        "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
        "Pourrait être un cas de rhume.\n"
        "Motif :\n"
        "Douleurs dans le bras droit."
    )

    doc1 = nlp(text)
    doc1.ents = doc1._.pollutions

    doc2 = nlp(text)
    doc2.ents = doc2._.section_titles

    docs = [doc1, doc2]
    for i, doc in enumerate(docs):
        doc._.note_id = i + 1

    brat2.docs2brat(docs)
