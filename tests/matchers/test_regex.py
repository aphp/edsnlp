import re

import pytest
from helpers import make_nlp
from pytest import mark

from edsnlp.matchers.regex import RegexMatcher, create_span
from edsnlp.matchers.utils import get_text


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


@mark.parametrize(
    "pattern, txt, span_from_group, result",
    [
        (
            r"match1 (?:group1|(group2))",  # pattern
            "It is a match1 group1",  # txt
            True,  # span_from_group
            "match1 group1",  # result
        ),
        (
            r"match1 (?:group1|(group2))",
            "It is a match1 group1",
            False,
            "match1 group1",
        ),
        (
            r"match1 (?:group1|(group2))",
            "It is a match1 group2",
            True,
            "group2",
        ),
        (
            r"match1 (?:group1|(group2))",
            "It is a match1 group2",
            False,
            "match1 group2",
        ),
    ],
)
def test_regex_with_groups(blank_nlp, pattern, txt, span_from_group, result):
    doc = blank_nlp(txt)
    matcher = RegexMatcher(span_from_group=span_from_group)
    matcher.add("test", [pattern])
    match = list(matcher(doc, as_spans=True))[0].text
    assert match == result


def test_regex_with_norm(blank_nlp):
    blank_nlp.add_pipe("pollution")

    text = "pneumopathie à NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNB coronavirus"

    doc = blank_nlp(text)

    matcher = RegexMatcher(ignore_excluded=True)
    matcher.add("test", ["pneumopathie à coronavirus"])

    match = list(matcher(doc, as_spans=True))[0]
    assert match.text == text
    assert match._.normalized_variant == "pneumopathie à coronavirus"


def test_regex_with_norm_on_span(blank_nlp):
    blank_nlp.add_pipe("pollution")

    text = (
        "le patient a une pneumopathie à NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNB"
        " coronavirus"
    )

    for offset in (0, 2):
        doc = blank_nlp(text)[offset:]

        matcher = RegexMatcher(ignore_excluded=True)
        matcher.add("test", ["pneumopathie à coronavirus"])

        match = list(matcher(doc, as_spans=True))[0]
        assert (
            match.text
            == "pneumopathie à NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNB coronavirus"
        )
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


def test_norm_alignment(blank_nlp):
    text = "test " + "bla… " * 4 + "test " + "bla" * 10

    blank_nlp.add_pipe(
        "matcher", config=dict(regex=dict(test=r"\btest\b"), attr="NORM")
    )

    doc = blank_nlp(text)

    for ent in doc.ents:
        assert ent.text == "test"


@mark.parametrize(
    "leading_text",
    [
        "",
        "\n",
        "Test de non-pollution",
    ],
)
@mark.parametrize("leading_pollution", [True, False])
@mark.parametrize("pollution_within", [True, False])
@mark.parametrize("trailing_pollution", [True, False])
@mark.parametrize(
    "pollution",
    ["==================", "======= ======= =======", "Nnnnnnnnnnnnn nnnnnn nnnnnnnn"],
)
def text_get_text(
    blank_nlp,
    leading_text: str,
    leading_pollution: bool,
    pollution_within: bool,
    trailing_pollution: bool,
    pollution: str,
):
    if pollution_within:
        example = f"transplantation {pollution} cardiaque en 2000."
    else:
        example = "transplantation cardiaque en 2000."

    chunks = []

    if leading_text:
        chunks.append(leading_text)
    if leading_pollution:
        chunks.append(pollution)

    chunks.append(example)

    if trailing_pollution:
        chunks.append(pollution)

    text = " ".join(chunks)

    blank_nlp.add_pipe("eds.normalizer", config=dict(pollution=True))
    blank_nlp.add_pipe(
        "eds.matcher",
        config=dict(
            regex=dict(test="transplantation cardiaque"),
            attr="NORM",
            ignore_excluded=True,
        ),
    )
    doc = blank_nlp(text)

    clean = get_text(doc, attr="NORM", ignore_excluded=True)
    if leading_text:
        assert clean == f"{leading_text.lower()} transplantation cardiaque en 2000."
    else:
        assert clean == "transplantation cardiaque en 2000."

    assert doc.ents
    assert doc.ents[0][0].text == "transplantation"

    clean = get_text(doc.ents[0], attr="NORM", ignore_excluded=True)
    assert clean == "transplantation cardiaque"


def test_groupdict_as_spans(doc):
    matcher = RegexMatcher()

    matcher.add("test", [r"patient(?i:(?=.*(?P<cause>douleurs))?)"])

    [(span0, gd0), (span1, gd1)] = list(matcher.match_with_groupdict_as_spans(doc))
    assert span0.text == "patient"
    assert span1.text == "patient"
    assert len(gd0) == 1 and gd0["cause"].text == "douleurs"
    assert len(gd1) == 0


def test_regex_with_space(blank_nlp):
    blank_nlp.add_pipe("eds.spaces")

    text = "pneumopathie à      coronavirus"

    doc = blank_nlp(text)

    matcher = RegexMatcher(ignore_space_tokens=False)
    matcher.add("test", ["pneumopathie à coronavirus"])

    assert len(list(matcher(doc, as_spans=True))) == 0

    matcher = RegexMatcher(ignore_space_tokens=True)
    matcher.add("test", ["pneumopathie à coronavirus"])

    match = list(matcher(doc, as_spans=True))[0]
    assert match.text == text
    assert match._.normalized_variant == "pneumopathie à coronavirus"


@pytest.fixture(scope="session")
def doc2(lang):
    blank_nlp = make_nlp(lang)
    blank_nlp.add_pipe("eds.pollution")
    blank_nlp.add_pipe("eds.spaces")

    text = (
        "-----------------------------------------------------------------------\n"
        "La ………… valeur est NBNbWbWbNbWbNBNb de 24 / 30 milli\n"
        "grammes."
    )

    doc = blank_nlp(text)

    return doc


@mark.parametrize("ignore_excluded", [True, False])
@mark.parametrize("ignore_space_tokens", [True, False])
@mark.parametrize("attr", ["TEXT", "NORM"])
@mark.parametrize("full_doc", [True, False])
def test_create_span(
    doc2,
    ignore_excluded: bool,
    ignore_space_tokens: bool,
    attr: str,
    full_doc: bool,
):
    sent = list(doc2.sents)[1]
    doclike = doc2 if full_doc else sent

    matched_text = get_text(
        doclike,
        attr=attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
    )
    clean_tokens = [
        t
        for t in doclike
        if not (
            (ignore_excluded and t.tag_ == "EXCLUDED")
            or (ignore_space_tokens and t.tag_ == "SPACE")
        )
    ]
    filtered_original = doc2[clean_tokens[0].i : clean_tokens[-1].i + 1].text
    for pattern, result, alignment_mode in [
        (r"4 / 3", "24 / 30", "expand"),
        (r"4 / 3", None, "strict"),
        (r"4 / 3", "/", "contract"),
        (r"24 / 30", "24 / 30", "expand"),
        (r"24 / 30", "24 / 30", "strict"),
        (r"24 / 30", "24 / 30", "contract"),
        (r"24 / 30 milli\s?gra", "24 / 30 milli\ngrammes", "expand"),
        (r"24 / 30 milli\s?gra", None, "strict"),
        (r"24 / 30 milli\s?gra", "24 / 30 milli\n", "contract"),
        (r" 24 / 30 ", "24 / 30", "expand"),
        (r" 24 / 30 ", None, "strict"),
        (r" 24 / 30 ", "24 / 30", "contract"),
        (matched_text, filtered_original, "expand"),
        (matched_text, filtered_original, "contract"),
        (matched_text, filtered_original, "strict"),
        ("(?=4 / 3)", "24", "expand"),
        ("(?=4 / 3)", None, "contract"),  # spacy behavior, but it's not ideal
        ("(?=4 / 3)", None, "strict"),
        ("(?=24)", "", "expand"),
        ("(?=24)", None, "contract"),  # spacy behavior, but it's not ideal
        ("(?=24)", None, "strict"),
    ]:
        match = re.search(pattern, matched_text)
        span = create_span(
            doclike,
            start_char=match.start(),
            end_char=match.end(),
            key="value",
            attr=attr,
            alignment_mode=alignment_mode,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
        )
        assert (None if span is None else span.text) == result, (
            pattern,
            result,
            alignment_mode,
        )


def test_create_empty_span(blank_nlp):
    blank_nlp.add_pipe("eds.pollution")
    blank_nlp.add_pipe("eds.spaces")
    doc = blank_nlp("plan des addictions:\ntabac :0")

    span = create_span(
        doc[5:],
        0,
        0,
        "empty",
        attr="NORM",
        alignment_mode="expand",
        ignore_excluded=True,
        ignore_space_tokens=True,
    )
    assert span.start == 5 and span.end == 5


def test_empty_get_text(blank_nlp):
    blank_nlp.add_pipe("eds.pollution")
    blank_nlp.add_pipe("eds.spaces")
    doc = blank_nlp("==================================")
    clean = get_text(doc, attr="NORM", ignore_excluded=True, ignore_space_tokens=True)
    assert clean == ""


def test_ignore_space_tokens_and_newline(blank_nlp):
    # Fixed by
    # `text_parts[i - 1] += " "` snippet in
    # edsnlp.utils.doc_to_text.aggregate_tokens
    blank_nlp.add_pipe("eds.normalizer")
    blank_nlp.add_pipe(
        "eds.matcher",
        config=dict(
            regex=dict(test=r"pneumopathie a coronavirus"),
            attr="NORM",
            ignore_excluded=True,
            ignore_space_tokens=True,
        ),
    )
    text = "Il a une\npneumopathie à coronavirus"
    doc = blank_nlp(text)
    clean = get_text(doc, attr="NORM", ignore_space_tokens=True, ignore_excluded=True)
    assert clean == "il a une pneumopathie a coronavirus"
    assert [e.text for e in doc.ents] == ["pneumopathie à coronavirus"]
