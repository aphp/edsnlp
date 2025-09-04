from edsnlp.utils.examples import Entity, parse_example

text = """
RCP du <ent lf1=None lf2=None>18/12/2024</ent> : DUPONT Jean

Homme de 68 ans adressé en consultation d’oncologie pour prise en charge d’une tumeur du colon.
Antécédents : HTA, diabète de type 2, dyslipidémie, tabagisme actif (30 PA), alcoolisme chronique (60 g/jour).

Examen clinique : patient en bon état général, poids 80 kg, taille 1m75.


HISTOIRE DE LA MALADIE :
Lors du PET-CT (<ent lf1=None lf2=None>14/02/2024</ent>), des dépôts pathologiques ont été observés qui coïncidaient avec les résultats du scanner.
Le <ent lf1='Magnetic resonance imaging (procedure)' lf2=None>15/02/2024</ent>, une IRM a été réalisée pour évaluer l’extension de la tumeur.
Une colonoscopie a été réalisée le <ent lf1=None lf2='Biopsy (procedure)'>17/02/2024</ent> avec une biopsie d'adénopathie sous-carinale.
Une deuxième a été biopsié le <ent lf1=None lf2='Biopsy (procedure)'>18/02/2024</ent>. Les résultats de la biopsie ont confirmé un adénocarcinome du colon.
Il a été opéré le <ent lf1=None lf2=None>20/02/2024</ent>. L’examen anatomopathologique de la pièce opératoire a confirmé un adénocarcinome du colon stade IV avec métastases hépatiques et pulmonaires.
Trois mois après la fin du traitement de chimiothérapie (<ent lf1=None lf2=None>avril 2024</ent>), le patient a signalé une aggravation progressive des symptômes

CONCLUSION :  Adénocarcinome du colon stade IV avec métastases hépatiques et pulmonaires.
"""  # noqa: E501

examples = [
    text,
]


def test_contextual_qualifier(edsnlp_blank_nlp):
    edsnlp_blank_nlp.add_pipe("eds.dates")

    edsnlp_blank_nlp.add_pipe(
        "eds.contextual_qualifier",
        config=dict(
            span_getter="dates",
            patterns={
                "lf1": {
                    "Magnetic resonance imaging (procedure)": {
                        "terms": {"irm": ["IRM", "imagerie par résonance magnétique"]},
                        "regex": None,
                        "context_words": 0,
                        "context_sents": 1,
                        "attr": "TEXT",
                    }
                },
                "lf2": {
                    "Biopsy (procedure)": {
                        "regex": {"biopsy": ["biopsie", "biopsié"]},
                        "context_words": (10, 10),
                        "context_sents": 0,
                        "attr": "TEXT",
                    }
                },
            },
        ),
    )

    for text, entities in map(parse_example, examples):
        doc = edsnlp_blank_nlp(text)

        dates = doc.spans["dates"]

        assert len(dates) == len(entities)

        for ent, entity in zip(dates, entities):
            entity: Entity
            assert ent.text == text[entity.start_char : entity.end_char]
            assert ent._.lf1 == entity.modifiers_dict.get("lf1")
            assert ent._.lf2 == entity.modifiers_dict.get("lf2")
