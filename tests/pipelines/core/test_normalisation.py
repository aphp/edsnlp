import re

import pytest
from pytest import fixture
from spacy.tokens import Doc

from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.core.normalizer.accents.patterns import accents
from edsnlp.pipelines.core.normalizer.pollution.patterns import pollution
from edsnlp.pipelines.core.normalizer.quotes.patterns import quotes_and_apostrophes
from edsnlp.utils.regex_utils import compile_regex


@fixture
def text():
    return "L'aïeul ʺnˊest pas malade”, écrit-il. Fièvre NBNbWbWbNbWbNB jaune."


@fixture
def doc(nlp, text):
    return nlp(text)


def test_full_normalization(doc):
    norm = get_text(doc, attr="NORM", ignore_excluded=True)
    assert doc[1].norm_ == "aieul"
    assert norm == "l'aieul \"n'est pas malade\", ecrit-il. fievre jaune."


@fixture
def nlp_factory(blank_nlp):
    def f(a=False, lc=False, q=False, p=False):
        if a:
            a = dict(accents=accents)
        if q:
            q = dict(quotes=quotes_and_apostrophes)
        if p:
            p = dict(pollution=pollution)

        blank_nlp.add_pipe(
            "normalizer",
            config=dict(
                accents=a,
                lowercase=lc,
                quotes=q,
                pollution=p,
            ),
        )
        return blank_nlp

    return f


def test_normalization_accents(nlp_factory, text):
    nlp = nlp_factory(a=True)
    doc = nlp(text)

    norm = get_text(doc, attr="NORM", ignore_excluded=True)

    assert norm == "L'aieul ʺnˊest pas malade”, ecrit-il. Fievre NBNbWbWbNbWbNB jaune."


def test_normalization_spaces(nlp_factory, text):
    nlp = nlp_factory(a=True)
    doc = nlp("Phrase    avec des espaces \n et un retour à la ligne")

    tags = [t.tag_ for t in doc]
    assert tags == ["", "SPACE", "", "", "", "SPACE", "", "", "", "", "", ""]


def test_normalization_quotes(nlp_factory, text):
    nlp = nlp_factory(q=True)
    doc = nlp(text)

    norm = get_text(doc, attr="NORM", ignore_excluded=True)

    assert (
        norm == "L'aïeul \"n'est pas malade\", écrit-il. Fièvre NBNbWbWbNbWbNB jaune."
    )


def test_normalization_lowercase(nlp_factory, text):
    nlp = nlp_factory(lc=True)
    doc = nlp(text)

    norm = get_text(doc, attr="NORM", ignore_excluded=True)

    assert norm.startswith("l'aïeul")


def test_normalization_pollution(nlp_factory, text):
    nlp = nlp_factory(p=True)
    doc = nlp(text)

    norm = get_text(doc, attr="NORM", ignore_excluded=True)

    assert norm == "L'aïeul ʺnˊest pas malade”, écrit-il. Fièvre jaune."

    text2 = "Le jour de \n"
    text2 += "2/2Pat : <NOM> <Prenom> le <date> IPP <ipp> Intitulé RCP"
    text2 += " : Urologie HMN le <date>\nRéunion de Concertation"
    text2 += " Pluridisciplinaire\nHôpital Henri Mondor"
    text2_expected = (
        "Le jour de \nRéunion de Concertation Pluridisciplinaire\nHôpital Henri Mondor"
    )

    text3 = "Le jour de \n"
    text3 += "3/5CRH service ABC HC SOINS INTENSIFS CARDIOLOGIE - CARDIOLOGIE-2EME"
    text3 += " ETAGE-B    Pat.: Prenom NOM | M | 13/10/1789 | 8012345678 | xxxxxxxx \n"
    text3 += "consultation"
    text3_expected = "Le jour de \nconsultation"

    examples = [(text2, text2_expected), (text3, text3_expected)]

    for example, expected in examples:
        doc = nlp(example)
        norm = get_text(doc, attr="NORM", ignore_excluded=True)
        assert norm == expected


def test_normalization_intraword_breaks(nlp_factory, lang):
    nlp = nlp_factory(p=True)
    example = "Le patient a un diab-\nète de type II."
    expected = "Le patient a un diabète de type II."
    doc = nlp(example)
    norm = get_text(doc, attr="NORM", ignore_excluded=True)
    if lang != "eds":
        pytest.xfail("This test is expected to fail when EDS's language isn't used")
    assert norm == expected


@pytest.mark.parametrize(
    "name,text,expected",
    [
        (
            "information",
            "=====\nInformation aux patients https://recherche.aphp.fr/eds/droit-opposition",
            [
                "=====\nInformation aux patients https://recherche.aphp.fr/eds/droit-opposition"
            ],
        ),
        (
            "information",
            "=== Information aux patients https://recherche.aphp.fr/eds/droit-opposition",
            ["Information aux patients https://recherche.aphp.fr/eds/droit-opposition"],
        ),
        (
            "biology",
            "  !!!Na | 140\nK ¦ 4\n  CRP | 2\nConclusion",
            ["Na | 140\nK ¦ 4\n", "CRP | 2\n"],
        ),
        ("biology", "|| Na\nNa | 140", []),
        ("biology", "Na | 140\r\nK | 4\r\n", ["Na | 140\r\nK | 4\r\n"]),
        ("biology", "¹ Na | 140\n\u0301K | 4\n", ["¹ Na | 140\n", "K | 4\n"]),
        (
            "coding",
            "head (1) A12 x B34 tail (2) C56 x D78",
            ["head (1) A12 x B34", " tail (2) C56 x D78"],
        ),
        (
            "coding",
            "head (1) A12\nNext (2) C56\n",
            ["head (1) A12\n", "Next (2) C56\n"],
        ),
        ("coding", "head (1) A12", []),
        (
            "footer",
            "1/2 heading\nPat exemple\nCourrier valide\nSuite",
            ["1/2 heading\nPat exemple\nCourrier valide"],
        ),
        (
            "footer",
            "prefix 01/02/2020 x 03/04/2020 8012345678 suffix\nSuite",
            ["prefix 01/02/2020 x 03/04/2020 8012345678 suffix"],
        ),
        ("footer", "8012345678 01/02/2020", []),
        ("footer", "²01/02/2020 8012345678", []),
        (
            "footer",
            "\u030101/02/2020\u0301 8012345678",
            ["\u030101/02/2020\u0301 8012345678"],
        ),
        (
            "footer",
            "ımprımé\x1cle\x1f01/02/2020 1/2 pat 03/04/2020",
            ["ımprımé\x1cle\x1f01/02/2020 1/2 pat 03/04/2020"],
        ),
        (
            "footer",
            "prefix imprimé le 01/02/2020 1/2\npat 03/04/2020 suffix",
            ["imprimé le 01/02/2020 1/2\npat 03/04/2020"],
        ),
        (
            "footer",
            "imprimé le 01/02/2020 1/2 pat 03/04/2020\npat 05/06/2020",
            ["imprimé le 01/02/2020 1/2 pat 03/04/2020\npat 05/06/2020"],
        ),
        (
            "footer",
            "imprimé le 01/02/2020 1/2 pat 03/04/2020\npat sans date",
            ["imprimé le 01/02/2020 1/2 pat 03/04/2020"],
        ),
        (
            "footer",
            "prefix imprimé\nle\n01/02/2020 1/2 pat 03/04/2020 suffix",
            ["imprimé\nle\n01/02/2020 1/2 pat 03/04/2020"],
        ),
        (
            "web",
            "x@y@z.fr next user@example.org",
            ["x@y@z.fr", "user@example.org", "x@y@z.fr", "user@example.org"],
        ),
        ("web", "x.fr.comsuffix\nexample.org", ["x.fr.com", "example.org"]),
    ],
)
def test_pollution_spans(name, text, expected):
    patterns = pollution[name]
    patterns = patterns if isinstance(patterns, list) else [patterns]
    assert [
        match.group()
        for pattern in patterns
        for match in compile_regex(pattern, re.MULTILINE).finditer(text)
    ] == expected


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Ò** lmÍ6AÓ" * 10000, id="reported"),
        pytest.param("X" * 100000, id="no-spaces"),
        pytest.param("8012345678 " + "01/02/2020 " * 10000, id="dates-after-ipp"),
        pytest.param("imprimé le 01/02/2020 " * 5000, id="incomplete-footers"),
    ],
)
def test_normalization_long_line(nlp_factory, text):
    nlp = nlp_factory(p=True)
    doc = nlp(text)
    assert not doc.spans["pollutions"]
    assert not any(token._.excluded for token in doc)


@pytest.mark.timeout(10)
def test_normalization_separator_line(nlp_factory):
    # Use a tokenized document to isolate pollution matching from tokenization
    nlp = nlp_factory(p=True)
    doc = nlp(Doc(nlp.vocab, words=["=" * 100000], spaces=[False]))
    assert [(span.label_, span.text) for span in doc.spans["pollutions"]] == [
        ("bars", doc.text)
    ]
    assert all(token._.excluded for token in doc)
