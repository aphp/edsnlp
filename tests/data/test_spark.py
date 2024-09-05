import edsnlp


def test_read_write(blank_nlp, text, df_notes_pyspark):
    # line below is just to mix params to avoid running too many tests
    shuffle = "dataset" if blank_nlp.lang == "eds" else False

    reader = edsnlp.data.from_spark(
        df_notes_pyspark,
        converter="omop",
        nlp=blank_nlp,
        shuffle=shuffle,
    ).set_processing(backend="simple")
    doc = list(reader)[0]
    assert doc.text == text

    blank_nlp.add_pipe("eds.matcher", config={"terms": {"douleur": ["douleurs"]}})
    blank_nlp.add_pipe("eds.negation")
    docs = blank_nlp.pipe(reader)

    writer = edsnlp.data.to_spark(
        docs,
        converter="omop",
        span_attributes=["negation"],
        span_getter=["ents"],
    )
    res = writer.toPandas().to_dict(orient="records")
    assert len(res) == 20
    assert sum(len(r["entities"]) for r in res) == 20
