def test_pipelines(doc):
    assert len(doc.ents) == 3
    patient, _, anomalie = doc.ents

    assert not patient._.negation
    assert anomalie._.negation

    assert not doc[0]._.history


def test_import_all():
    import edsnlp.pipes

    for name in dir(edsnlp.pipes):
        if not name.startswith("_"):
            getattr(edsnlp.pipes, name)
