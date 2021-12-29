import pytest

from edsnlp.matchers.regex import RegexMatcher


def test_regex(doc):
    matcher = RegexMatcher()

    matcher.add("test", [r"test"])
    matcher.remove("test")

    matcher.add("patient", [r"patient"])

    matches = matcher(doc, as_spans=False)

    for _, start, end in matcher(doc, as_spans=False):
        assert len(doc[start:end])

    matches = matcher(doc[:10])

    assert list(matches)


def test_regex_with_norm(blank_nlp):
    blank_nlp.add_pipe("pollution")

    text = "pneumopathie à NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNB coronavirus"

    doc = blank_nlp(text)

    matcher = RegexMatcher(ignore_excluded=True)
    matcher.add("test", ["pneumopathie à coronavirus"])

    match = list(matcher(doc, as_spans=True))[0]
    assert match.text == text
    assert match._.normalized_variant == "pneumopathie à coronavirus"


def test_offset(blank_nlp):

    text = "Ceci est un test de matching"

    doc = blank_nlp(text)
    pattern = "matching"

    matcher = RegexMatcher(attr="TEXT")

    matcher.add("test", [pattern])

    for _, start, end in matcher(doc):
        assert doc[start:end].text == pattern

    for span in matcher(doc, as_spans=True):
        span.text == pattern

    for _, start, end in matcher(doc[2:]):
        assert doc[2:][start:end].text == pattern

    for span in matcher(doc[2:], as_spans=True):
        span.text == pattern


def test_remove():

    matcher = RegexMatcher(attr="TEXT")

    matcher.add("test", ["pattern"])
    matcher.add("test", ["pattern2"], attr="LOWER")

    assert len(matcher) == 1

    with pytest.raises(ValueError):
        matcher.remove("wrong_key")

    matcher.remove("test")

    assert len(matcher) == 0
