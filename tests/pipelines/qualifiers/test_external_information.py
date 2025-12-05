import datetime

import pandas as pd

import edsnlp
from edsnlp.utils.examples import Entity, parse_example

text = """
RCP du <ent lf1=None>18/12/2024</ent> : DUPONT Jean

Homme de 68 ans adressé en consultation d’oncologie pour prise en charge d’une tumeur du colon.
Antécédents : HTA, diabète de type 2, dyslipidémie, tabagisme actif (30 PA), alcoolisme chronique (60 g/jour).

Examen clinique : patient en bon état général, poids 80 kg, taille 1m75.


HISTOIRE DE LA MALADIE :
Lors du PET-CT (<ent lf1=None>14/02/2024</ent>), des dépôts pathologiques ont été observés qui coïncidaient avec les résultats du scanner.
Le <ent lf1='Magnetic resonance imaging (procedure)'>15/02/2024</ent>, une IRM a été réalisée pour évaluer l’extension de la tumeur.
Une colonoscopie a été réalisée le <ent lf1='Biopsy (procedure)' lf1='Colonoscopy (procedure)'>17/02/2024</ent> avec une biopsie d'adénopathie sous-carinale.
Une deuxième a été biopsié le <ent lf1=None>18/02/2024</ent>. Les résultats de la biopsie ont confirmé un adénocarcinome du colon.
Il a été opéré le <ent lf1=None>20/02/2024</ent>. L’examen anatomopathologique de la pièce opératoire a confirmé un adénocarcinome du colon stade IV avec métastases hépatiques et pulmonaires.
Trois mois après la fin du traitement de chimiothérapie (<ent lf1=None>avril 2024</ent>), le patient a signalé une aggravation progressive des symptômes

CONCLUSION :  Adénocarcinome du colon stade IV avec métastases hépatiques et pulmonaires.
"""  # noqa: E501

# Create context dates
# The elements under this attribute should be a list of dicts with keys value and class
context_dates = [
    {
        "value": datetime.datetime(2024, 2, 15),
        "class": "Magnetic resonance imaging (procedure)",
    },
    {"value": datetime.datetime(2024, 2, 17), "class": "Biopsy (procedure)"},
    {"value": datetime.datetime(2024, 2, 17), "class": "Colonoscopy (procedure)"},
]

examples = [
    (text, context_dates),
]


def test_external_information_qualifier(edsnlp_blank_nlp):
    edsnlp_blank_nlp.add_pipe("eds.dates")

    edsnlp_blank_nlp.add_pipe(
        "eds.external_information_qualifier",
        config=dict(
            span_getter="dates",
            external_information={
                "lf1": dict(
                    doc_attr="_.context_dates",
                    span_attribute="_.date.to_datetime()",
                    threshold=datetime.timedelta(days=0),
                )
            },
        ),
    )

    texts = [text for text, _ in examples]
    context_dates = [context_dates for _, context_dates in examples]
    for (text, entities), context in zip(map(parse_example, texts), context_dates):
        # Create a dataframe
        df = pd.DataFrame.from_records(
            [
                {
                    "person_id": 1,
                    "note_id": 1,
                    "note_text": text,
                    "context_dates": context,
                }
            ]
        )

        doc_iterator = edsnlp.data.from_pandas(
            df, converter="omop", doc_attributes=["context_dates"]
        )

        docs = list(edsnlp_blank_nlp.pipe(doc_iterator))

        doc = docs[0]

        dates = doc.spans["dates"]

        assert len(dates) == len(entities)

        for ent, entity in zip(dates, entities):
            entity: Entity
            assert ent.text == text[entity.start_char : entity.end_char]
            for e in entity.modifiers:
                value = e.value
                if value:
                    assert value in ent._.lf1
