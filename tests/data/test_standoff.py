import filecmp
import re
from itertools import islice
from os import listdir
from os.path import join
from pathlib import Path
from random import choice, randint, random
from string import ascii_letters, ascii_lowercase

import pytest

import edsnlp
from edsnlp.connectors.brat import BratConnector
from edsnlp.core import PipelineProtocol


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


# OLD BratConnector tests, deprecated


@pytest.fixture
def brat1(brat_folder) -> BratConnector:
    return BratConnector(brat_folder)


@pytest.fixture
def brat2(tmpdir) -> BratConnector:
    return BratConnector(tmpdir)


@pytest.fixture
def brat_importer():
    brat_dir = Path(__file__).parent.parent.resolve() / "resources" / "brat_data"
    return BratConnector(str(brat_dir), bool_attributes=["bool flag 0"])


@pytest.fixture
def brat_exporter(tmpdir):
    return BratConnector(tmpdir, attributes=["etat", "assertion", "bool flag 0"])


def test_empty_brat(brat2: BratConnector, blank_nlp: PipelineProtocol):
    with pytest.raises(AssertionError):
        brat2.brat2docs(blank_nlp)


def test_brat2brat(
    brat1: BratConnector, brat2: BratConnector, blank_nlp: PipelineProtocol
):
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


def assert_doc_read(doc):
    assert doc._.note_id == "subfolder/doc-1"

    attrs = ("etat", "assertion", "bool flag 0")
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
            (6, 7, "douleurs", (None, None, False)),
            (7, 11, "dans le bras droit", (None, None, False)),
            (17, 21, "problème \nde locomotion", (None, "absent", True)),
            (25, 26, "AVC", ("passé", "non-associé", False)),
            (35, 36, "rhume", ("présent", "hypothétique", False)),
            (45, 46, "rhume", ("présent", "hypothétique", False)),
            (51, 52, "Douleurs", (None, None, False)),
            (52, 56, "dans le bras droit", (None, None, False)),
            (68, 69, "anomalie", (None, "absent", False)),
        ],
        "anatomie": [
            (9, 11, "bras droit", (None, None, False)),
            (54, 56, "bras droit", (None, None, False)),
        ],
        "localisation": [
            (7, 11, "dans le bras droit", (None, None, False)),
            (52, 56, "dans le bras droit", (None, None, False)),
        ],
        "pathologie": [
            (17, 21, "problème \nde locomotion", (None, "absent", True)),
            (25, 26, "AVC", ("passé", "non-associé", False)),
            (35, 36, "rhume", ("présent", "hypothétique", False)),
            (45, 46, "rhume", ("présent", "hypothétique", False)),
        ],
        "sosy": [
            (6, 7, "douleurs", (None, None, False)),
            (51, 52, "Douleurs", (None, None, False)),
            (68, 69, "anomalie", (None, "absent", False)),
        ],
        "test label 0": [(68, 69, "anomalie", (None, "absent", False))],
    }


def assert_doc_write(exported_ann_text):
    assert exported_ann_text == (
        "T1	sosy 30 38	douleurs\n"
        "A1	etat T1 test\n"
        "T2	localisation 39 57	dans le bras droit\n"
        "T3	anatomie 47 57	bras droit\n"
        "T4	pathologie 75 83;85 98	problème de locomotion\n"
        "A2	assertion T4 absent\n"
        "A3	bool flag 0 T4\n"
        "T5	pathologie 114 117	AVC\n"
        "A4	etat T5 passé\n"
        "A5	assertion T5 non-associé\n"
        "T6	pathologie 159 164	rhume\n"
        "A6	etat T6 présent\n"
        "A7	assertion T6 hypothétique\n"
        "T7	pathologie 291 296	rhume\n"
        "A8	etat T7 présent\n"
        "A9	assertion T7 hypothétique\n"
        "T8	sosy 306 314	Douleurs\n"
        "T9	localisation 315 333	dans le bras droit\n"
        "T10	anatomie 323 333	bras droit\n"
        "T11	sosy 378 386	anomalie\n"
        "A10	assertion T11 absent\n"
        "T12	test label 0 378 386	anomalie\n"
        "A11	assertion T12 absent\n"
    )


def test_brat(
    brat_importer: BratConnector,
    brat_exporter: BratConnector,
    blank_nlp: PipelineProtocol,
):
    doc = brat_importer.brat2docs(blank_nlp)[0]
    assert_doc_read(doc)
    doc.ents[0]._.etat = "test"

    brat_exporter.docs2brat([doc])
    with open(brat_exporter.directory / "subfolder" / "doc-1.ann") as f:
        exported_ann_text = f.read()

    assert_doc_write(exported_ann_text)


# New `edsnlp.data.read_standoff` and `edsnlp.data.write_standoff` tests


def test_read_to_standoff(blank_nlp, tmpdir):
    input_dir = Path(__file__).parent.parent.resolve() / "resources" / "brat_data"
    output_dir = Path(tmpdir)
    doc = list(
        edsnlp.data.read_standoff(
            input_dir,
            bool_attributes=["bool flag 0"],
            notes_as_span_attribute="cui",
        )
    )[0]
    assert_doc_read(doc)
    doc.ents[0]._.etat = "test"
    doc.ents[0]._.cui = "C0030193"

    edsnlp.data.write_standoff(
        [doc],
        output_dir,
        span_attributes=["etat", "assertion", "bool flag 0"],
        span_getter=[
            "ents",
            "sosy",
            "localisation",
            "anatomie",
            "pathologie",
            "test label 0",
        ],
    )

    with open(output_dir / "subfolder" / "doc-1.ann") as f:
        exported_ann_text = f.read()

    assert_doc_write(exported_ann_text)


@pytest.mark.parametrize("num_cpu_workers", [0, 2])
def test_read_shuffle_loop(num_cpu_workers: int):
    notes = (
        edsnlp.data.read_standoff(
            Path(__file__).parent.parent.resolve() / "resources" / "brat_data",
            shuffle="dataset",
            keep_txt_only_docs=True,
            seed=42,
            loop=True,
        )
        .map(lambda x: x._.note_id)
        .set_processing(num_cpu_workers=num_cpu_workers)
    )
    notes = list(islice(notes, 6))
    assert notes == [
        "subfolder/doc-2",
        "subfolder/doc-1",
        "subfolder/doc-3",
        "subfolder/doc-3",
        "subfolder/doc-2",
        "subfolder/doc-1",
    ]
