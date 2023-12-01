import pytest

from edsnlp.matchers.phrase import EDSPhraseMatcher


def test_eds_phrase_matcher(doc, nlp):
    matcher = EDSPhraseMatcher(nlp.vocab, attr="TEXT")

    matcher.add("test", list(nlp.pipe(["test"])))
    matcher.remove("test")

    matcher.add("patient", list(nlp.pipe(["patient"])))

    matches = matcher(doc, as_spans=False)

    assert list(matches)

    matches = matcher(doc[:10])

    assert list(matches)


def test_offset(blank_nlp):

    text = "Ceci est un test de matching"

    doc = blank_nlp(text)
    pattern = blank_nlp("matching")

    matcher = EDSPhraseMatcher(blank_nlp.vocab, attr="TEXT")

    matcher.add("test", [pattern])

    for _, start, end in matcher(doc):
        assert doc[start:end].text == pattern.text

    for span in matcher(doc, as_spans=True):
        span.text == pattern.text

    for _, start, end in matcher(doc[2:]):
        assert doc[start:end].text == pattern.text

    for span in matcher(doc[2:], as_spans=True):
        span.text == pattern.text


def test_remove(blank_nlp):

    pattern = blank_nlp("matching")
    pattern2 = blank_nlp("Ceci")

    matcher = EDSPhraseMatcher(blank_nlp.vocab, attr="TEXT")

    matcher.add("test", [pattern])
    matcher.add("test", [pattern2])

    assert len(matcher) == 1

    with pytest.raises(KeyError):
        matcher.remove("wrong_key")

    matcher.remove("test")

    assert len(matcher) == 0
