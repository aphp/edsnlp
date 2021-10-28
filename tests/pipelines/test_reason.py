import pytest
import spacy

import edsnlp.components

text = """COMPTE RENDU D'HOSPITALISATION du 11/07/2018 au 12/07/2018
MOTIF D'HOSPITALISATION
Monsieur Dupont Jean Michel, de sexe masculin, âgée de 39 ans, née le 23/11/1978, a été
hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme.

ANTÉCÉDENTS
Antécédents médicaux :
Premier épisode d'asthme en mai 2018."""


def test_reason():
    nlp = spacy.blank("fr")
    # Extraction d'entités nommées
    nlp.add_pipe(
        "matcher",
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

    nlp.add_pipe("normalizer")
    nlp.add_pipe("reason", config=dict(use_sections=True))

    doc = nlp(text)

    reason = doc.spans["reasons"][0]

    entities = reason._.ents_reason

    assert entities[0].label_ == "respiratoire"
