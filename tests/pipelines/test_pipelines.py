def test_pipelines(doc):
    assert len(doc.ents) == 2
    patient, anomalie = doc.ents

    assert not patient._.negated
    assert anomalie._.negated
