from itertools import islice
from pathlib import Path

import pytest
from typing_extensions import Literal

import edsnlp


@pytest.mark.parametrize("num_cpu_workers", [0, 2])
@pytest.mark.parametrize("shuffle", ["dataset"])
def test_read_shuffle_loop(
    num_cpu_workers: int,
    shuffle: Literal["dataset", "fragment"],
):
    input_file = (
        Path(__file__).parent.parent.resolve() / "training" / "rhapsodie_sample.conllu"
    )
    notes = edsnlp.data.read_conll(
        input_file,
        shuffle=shuffle,
        seed=42,
        loop=True,
    ).set_processing(num_cpu_workers=num_cpu_workers)
    notes = list(islice(notes, 6))
    assert len(notes) == 6
    # 32	ce	ce	PRON	_	Gender=Masc|Number=Sing|Person=3|PronType=Dem	30	obl:arg	_	_  # noqa: E501
    word_attrs = {
        "text": "ce",
        "lemma_": "ce",
        "pos_": "PRON",
        "dep_": "obl:arg",
        "morph": "Gender=Masc|Number=Sing|Person=3|PronType=Dem",
        "head": "profité",
    }
    word = notes[0][31]
    for attr, val in word_attrs.items():
        assert str(getattr(word, attr)) == val
