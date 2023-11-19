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
