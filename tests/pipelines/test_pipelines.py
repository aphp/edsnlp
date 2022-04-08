def test_pipelines(doc):
    assert len(doc.ents) == 3
    patient, _, anomalie = doc.ents

    assert not patient._.negation
    assert anomalie._.negation

    assert not doc[0]._.history
