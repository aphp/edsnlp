import spacy

from edsnlp.pipelines.sentences import SentenceSegmenter, terms
from spacy.pipeline.sentencizer import Sentencizer

text = (
    "Le patient est admis pour des douleurs dans le bras droit. mais n'a pas de problème de locomotion. \n"
    "Historique d'AVC dans la famille\n"
    "mais ne semble pas en être un\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit."
)


def test_sentences(nlp):
    nlp_blank = spacy.blank("fr")
    sentencizer = Sentencizer()
    segmenter = SentenceSegmenter(punct_chars=terms.punctuation)

    doc = nlp(text)
    doc_blank = nlp_blank(text)
    assert len(list(sentencizer(doc_blank).sents)) == 4

    doc = nlp(text)
    assert len(list(segmenter(doc).sents)) == 6

    segmenter(nlp_blank(""))
