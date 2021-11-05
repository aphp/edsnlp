from edsnlp.matchers.phrase import EDSPhraseMatcher


def test_eds_phrase_matcher(doc, nlp):
    matcher = EDSPhraseMatcher(nlp.vocab, attr="CUSTOM_NORM")

    matcher.add("test", list(nlp.pipe(["test"])))
    matcher.remove("test")

    matcher.add("patient", list(nlp.pipe(["patient"])))

    matches = matcher(doc, as_spans=False)

    assert list(matches)

    matches = matcher(doc[:10])

    assert list(matches)
