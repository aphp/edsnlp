import spacy
from pytest import mark

text = """COMPTE RENDU D'HOSPITALISATION du 11/07/2018 au 12/07/2018
MOTIF D'HOSPITALISATION
Monsieur Dupont Jean Michel, de sexe masculin, âgée de 39 ans, née le 23/11/1978,
a été hospitalisé du 11/08/2019 au 17/08/2019 pour une quinte de toux.

ANTÉCÉDENTS
Antécédents médicaux :
Premier épisode: il a été hospitalisé pour asthme en mai 2018."""


@mark.parametrize("use_sections", [True, False])
def test_reason(lang, use_sections):
    nlp = spacy.blank(lang)
    # Extraction d'entités nommées
    nlp.add_pipe(
        "eds.matcher",
        config=dict(
            terms=dict(
                respiratoire=[
                    "asthmatique",
                    "asthme",
                    "toux",
                ]
            )
        ),
    )
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.reason", config=dict(use_sections=use_sections))
    nlp.remove_pipe("eds.reason")
    nlp.add_pipe("eds.sections")
    nlp.add_pipe("eds.reason", config=dict(use_sections=use_sections))

    doc = nlp(text)
    reason = doc.spans["reasons"][0]
    entities = reason._.ents_reason

    assert entities[0].label_ == "respiratoire"
    assert reason._.is_reason
    assert doc.ents[1]._.is_reason is not use_sections
