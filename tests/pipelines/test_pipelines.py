from edsnlp.processing import pipe, parallel_pipe

import pytest


def test_pipelines(doc):
    assert len(doc.ents) == 2
    patient, anomalie = doc.ents

    assert patient._.date is None

    assert not patient._.negated
    assert anomalie._.negated
