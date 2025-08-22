import sys

import pytest

import edsnlp


def test_pipelines(doc):
    assert len(doc.ents) == 3
    patient, _, anomalie = doc.ents

    assert not patient._.negation
    assert anomalie._.negation

    assert not doc[0]._.history


def is_openai_3_7(e):
    return (
        "openai" in str(e)
        and sys.version_info.major == 3
        and sys.version_info.minor == 7
    )


def test_import_all():
    import edsnlp.pipes

    for name in dir(edsnlp.pipes):
        if not name.startswith("_") and "endlines" not in name:
            try:
                getattr(edsnlp.pipes, name)
            except (ImportError, AttributeError) as e:
                if "torch" in str(e):
                    pass
                if is_openai_3_7(e):
                    # Skip tests for OpenAI using python 3.7
                    pass


def test_non_existing_pipe():
    with pytest.raises(AttributeError) as e:
        getattr(edsnlp.pipes, "non_existing_pipe")

    assert str(e.value) == "module edsnlp.pipes has no attribute non_existing_pipe"
