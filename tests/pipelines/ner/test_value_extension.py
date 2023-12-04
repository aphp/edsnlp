import pytest
import spacy
from spacy.tokens import Span
from spacy.tokens.underscore import Underscore

import edsnlp


def test_warn_value_extension():
    old_value_extension = Underscore.span_extensions.pop("value", None)
    try:
        Underscore._extensions = {}
        Span.set_extension("value", getter=lambda span: "stuff")
        existing_nlp = spacy.blank("fr")
        with pytest.warns(UserWarning) as record:
            existing_nlp.add_pipe(
                "eds.terminology",
                name="test",
                config=dict(label="Any", terms={}),
            )

        assert any(
            "A Span extension 'value' already exists with a different getter"
            in str(r.message)
            for r in record
        )
    finally:
        Underscore.span_extensions.pop("value", None)
        if old_value_extension is not None:
            Underscore.span_extensions["value"] = old_value_extension


def test_value_extension():
    # From https://github.com/aphp/edsnlp/issues/220

    # Setting up a first pipeline
    existing_nlp = spacy.blank("fr")
    existing_nlp.add_pipe(
        "eds.terminology",
        name="test",
        config=dict(label="Any", terms={}),
    )

    # Setting up another custom pipeline somewhere else in the code
    nlp = edsnlp.blank("eds")
    text = "hello this is a test"
    doc = nlp(text)
    my_span = doc[0:3]
    my_span._.value = "CustomValue"

    assert my_span._.value == "CustomValue"
