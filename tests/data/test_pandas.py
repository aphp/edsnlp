from itertools import islice
from pathlib import Path

import pytest

import edsnlp


def test_read_write(blank_nlp, text, df_notes_pandas):
    reader = edsnlp.data.from_pandas(
        df_notes_pandas,
        converter="omop",
        nlp=blank_nlp,
    ).set_processing(backend="simple")
    doc = list(reader)[0]
    assert doc.text == text

    blank_nlp.add_pipe("eds.matcher", config={"terms": {"douleur": ["douleurs"]}})
    blank_nlp.add_pipe("eds.negation")
    docs = reader.map_pipeline(blank_nlp)

    writer = docs.to_pandas(
        converter="omop",
        span_attributes=["negation"],
        span_getter=["ents"],
    )
    res = writer.to_dict(orient="records")
    assert len(res) == 20
    assert sum(len(r["entities"]) for r in res) == 20


@pytest.mark.parametrize("num_cpu_workers", [0, 2])
def test_read_shuffle_loop(num_cpu_workers: int):
    import pandas as pd

    data = pd.read_parquet(
        Path(__file__).parent.parent.resolve() / "resources" / "docs.parquet"
    )
    notes = (
        edsnlp.data.from_pandas(
            data,
            shuffle="dataset",
            seed=42,
            loop=True,
        )
        .map(lambda x: x["note_id"])
        .set_processing(num_cpu_workers=num_cpu_workers)
    )
    notes = list(islice(notes, 6))
    assert notes == [
        "subfolder/doc-3",
        "subfolder/doc-2",
        "subfolder/doc-1",
        "subfolder/doc-3",
        "subfolder/doc-1",
        "subfolder/doc-2",
    ]
