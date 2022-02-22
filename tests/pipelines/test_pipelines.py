def test_pipelines(doc):
    assert len(doc.ents) == 3
    patient, _, anomalie = doc.ents

    assert patient._.date == "????-??-??"

    assert not patient._.negation
    assert anomalie._.negation

    assert doc[0]._.antecedents_ == "CURRENT"
