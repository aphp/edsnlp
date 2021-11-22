import spacy
from pytest import mark
from spacy.pipeline.sentencizer import Sentencizer
from spacy.tokens import Token

from edsnlp.pipelines.sentences import SentenceSegmenter, terms

text = (
    "Le patient est admis pour des douleurs dans le bras droit. mais n'a pas de problème de locomotion. \n"
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
    segmenter = SentenceSegmenter(punct_chars=terms.punctuation, use_endlines=True)

    doc = nlp(text)
    doc_blank = nlp_blank(text)

    if endlines:
        if not Token.has_extension("end_line"):
            Token.set_extension("end_line", default=True)

        doc_blank[28]._.end_line = False

    doc_blank = segmenter(doc_blank)

    if endlines:
        assert len(list(doc_blank.sents)) == 6
    else:
        assert len(list(doc_blank.sents)) == 7
    assert len(list(sentencizer(doc).sents)) == 7

    segmenter(nlp_blank(""))
