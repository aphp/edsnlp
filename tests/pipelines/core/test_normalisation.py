from pytest import fixture

from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.core.normalizer.accents.patterns import accents
from edsnlp.pipelines.core.normalizer.pollution.patterns import pollution
from edsnlp.pipelines.core.normalizer.quotes.patterns import quotes_and_apostrophes


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
