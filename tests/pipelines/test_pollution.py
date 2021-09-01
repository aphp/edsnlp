def test_pollution_detection(doc):
    assert [token for token in doc if token._.pollution]
    assert len(doc._.clean_) < len(doc.text)


def test_pollution_alignment(doc):
    clean_extraction = doc._.clean_[165:181]

    # Testing realignment
    assert clean_extraction == doc._.char_clean_span(165, 181).text
