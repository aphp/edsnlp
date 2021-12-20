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
