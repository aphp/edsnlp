def test_pipelines(doc):
    assert len(doc.ents) == 3
    patient, _, anomalie = doc.ents

    assert patient._.date == "????-??-??"

    assert not patient._.negated
    assert anomalie._.negated

    assert doc[0]._.antecedent_ == "NOTSET"
