from pytest import fixture

from edsnlp.pipelines.normalizer.pollution.terms import pollution
from edsnlp.pipelines.normalizer.quotes.terms import quotes_and_apostrophes
from edsnlp.pipelines.normalizer.terms import accents
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
    norm = doc._.normalized.text
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

    norm = doc._.normalized.text

    assert (
        norm == "Le patient ʺnˊest pas malade”, ecrit-il. Fievre NBNbWbWbNbWbNB jaune."
    )


def test_normalization_quotes(nlp_factory, text):

    nlp = nlp_factory(q=True)
    doc = nlp(text)

    norm = doc._.normalized.text

    assert (
        norm == 'Le patient "n\'est pas malade", écrit-il. Fièvre NBNbWbWbNbWbNB jaune.'
    )


def test_normalization_lowercase(nlp_factory, text):

    nlp = nlp_factory(lc=True)
    doc = nlp(text)

    norm = doc._.normalized.text

    assert (
        norm == "le patient ʺnˊest pas malade”, écrit-il. fièvre nbnbwbwbnbwbnb jaune."
    )


def test_normalization_pollution(nlp_factory, text):

    nlp = nlp_factory(p=True)
    doc = nlp(text)

    norm = doc._.normalized.text

    assert norm == "Le patient ʺnˊest pas malade”, écrit-il. Fièvre jaune."
