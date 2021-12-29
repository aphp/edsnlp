from pytest import fixture

from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.normalizer.accents.terms import accents
from edsnlp.pipelines.normalizer.pollution.terms import pollution
from edsnlp.pipelines.normalizer.quotes.terms import quotes_and_apostrophes
from edsnlp.pipelines.normalizer.utils import replace


@fixture
def text():
    return "Le patient ʺnˊest pas malade”, écrit-il. Fièvre NBNbWbWbNbWbNB jaune."


@fixture
def doc(nlp, text):
    return nlp(text)


def test_replace():
    text = "üîéè"
    assert replace(text, accents) == "uiee"


def test_full_normalization(doc):
    norm = get_text(doc, attr="NORM", ignore_excluded=True)
    assert norm == 'le patient "n\'est pas malade", ecrit-il. fievre jaune.'


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

    assert (
        norm == "Le patient ʺnˊest pas malade”, ecrit-il. Fievre NBNbWbWbNbWbNB jaune."
    )


def test_normalization_quotes(nlp_factory, text):

    nlp = nlp_factory(q=True)
    doc = nlp(text)

    norm = get_text(doc, attr="NORM", ignore_excluded=True)

    assert (
        norm == 'Le patient "n\'est pas malade", écrit-il. Fièvre NBNbWbWbNbWbNB jaune.'
    )


def test_normalization_lowercase(nlp_factory, text):

    nlp = nlp_factory(lc=True)
    doc = nlp(text)

    norm = get_text(doc, attr="NORM", ignore_excluded=True)

    assert norm.startswith("le patient")


def test_normalization_pollution(nlp_factory, text):

    nlp = nlp_factory(p=True)
    doc = nlp(text)

    norm = get_text(doc, attr="NORM", ignore_excluded=True)

    assert norm == "Le patient ʺnˊest pas malade”, écrit-il. Fièvre jaune."
