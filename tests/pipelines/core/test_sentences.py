import spacy
from pytest import mark
from spacy.pipeline.sentencizer import Sentencizer

from edsnlp.pipelines.core.sentences import SentenceSegmenter, terms

text = (
    "Le patient est admis pour des douleurs dans le bras droit. "
    "mais n'a pas de problème de locomotion. \n"
    "Historique d'AVC dans la famille\n"
    "Mais ne semble pas en être un\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit."
)


@mark.parametrize("endlines", [True, False])
def test_sentences(nlp, endlines):
    nlp_blank = spacy.blank("fr")

    sentencizer = Sentencizer()
    segmenter = SentenceSegmenter(
        nlp.vocab, punct_chars=terms.punctuation, use_endlines=True
    )

    doc = nlp(text)
    doc_blank = nlp_blank(text)

    if endlines:
        doc_blank[28].tag_ = "EXCLUDED"

    doc_blank = segmenter(doc_blank)

    if endlines:
        assert len(list(doc_blank.sents)) == 6
    else:
        assert len(list(doc_blank.sents)) == 7
    assert len(list(sentencizer(doc).sents)) == 7

    segmenter(nlp_blank(""))


def test_false_positives(blank_nlp):

    false_positives = [
        "02.04.2018",
    ]

    for fp in false_positives:
        doc = blank_nlp(fp)
        assert len(list(doc.sents)) == 1
