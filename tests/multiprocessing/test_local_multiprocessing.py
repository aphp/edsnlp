import pytest

from edsnlp.multiprocessing import parallel_pipe, pipe


def test_pipeline_methods(nlp, df_notes):

    # 2nd method
    ents_2 = parallel_pipe(
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
    for doc in pipe(
        nlp,
        df_notes,
        text_col="note_text",
        context_cols=["note_id", "note_datetime"],
        progress_bar=False,
    ):
        if len(doc.ents) > 0:
            ents_1.extend(list(doc.ents))

    assert len(ents_1) >= 2 * len(df_notes)

    assert len(ents_2) >= 2 * len(df_notes)

    assert len(ents_1) == len(ents_2)


def test_parallel_pipe(nlp, df_notes):
    parallel_pipe(
        nlp,
        df_notes,
        chunksize=10,
        n_jobs=1,
        context_cols="note_id",
        text_col="note_text",
        progress_bar=True,
        return_df=True,
    )


def test_pipe(nlp, df_notes):

    with pytest.raises(ValueError):
        for doc in pipe(
            nlp,
            df_notes,
            text_col="note_text",
            context_cols=["note_ids"],
            progress_bar=False,
        ):
            pass
