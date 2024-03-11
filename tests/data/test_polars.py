import polars

import edsnlp


def test_read_write(blank_nlp, text, df_notes_pandas):
    df_notes_polars = polars.from_pandas(df_notes_pandas)
    reader = edsnlp.data.from_polars(
        df_notes_polars,
        converter="omop",
        nlp=blank_nlp,
    ).set_processing(backend="simple")
    doc = list(reader)[0]
    assert doc.text == text

    blank_nlp.add_pipe("eds.matcher", config={"terms": {"douleur": ["douleurs"]}})
    blank_nlp.add_pipe("eds.negation")
    docs = reader.map_pipeline(blank_nlp)

    writer: polars.DataFrame = docs.to_polars(
        converter="omop",
        span_attributes=["negation"],
        span_getter=["ents"],
    )
    res = writer.to_dicts()
    assert len(res) == 20
    assert sum(len(r["entities"]) for r in res) == 20
