import edsnlp.processing as nlprocess


def test_pipelines(doc):
    assert len(doc.ents) == 2
    patient, anomalie = doc.ents

    assert not patient._.negated
    assert anomalie._.negated


def test_pipeline_methods(nlp, df_notes):

    # 2nd method
    ents_2 = nlprocess.parallel_pipe(
        nlp,
        df_notes,
        chunksize=10,
        n_jobs=1,
        context_cols="note_id",
        text_col="note_text",
        progress_bar=True,
        return_df=False,
    )

    # 1st method
    ents_1 = []
    for doc in nlprocess.pipe(
        nlp,
        df_notes,
        text_col="note_text",
        context_cols=["note_id"],
        progress_bar=False,
    ):
        if len(doc.ents) > 0:
            ents_1.extend(list(doc.ents))

    assert len(ents_1) >= 2 * len(df_notes)

    assert len(ents_2) >= 2 * len(df_notes)

    assert len(ents_1) == len(ents_2)
