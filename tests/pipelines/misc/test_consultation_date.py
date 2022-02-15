from datetime import datetime

import pytest

from edsnlp.pipelines.misc.consultation_dates import factory  # noqa

TEXT = """
Références : AMO/AMO
Objet : Compte-Rendu de Consultation du 07/10/2018
Madame BEESLY Pamela, âgée de 45 ans, née le 05/10/1987, a été vue en consultation
dans le service de NCK CS RHUMATO.

####

Paris, le 24 janvier 2020

####

Document signé le 10/02/2020

"""

cons = dict(
    additionnal_params=dict(),
    result=[datetime(2018, 10, 7)],
)

cons_town = dict(
    additionnal_params=dict(town_mention=True),
    result=[datetime(2018, 10, 7), datetime(2020, 1, 24)],
)

cons_town_doc = dict(
    additionnal_params=dict(town_mention=True, document_date_mention=True),
    result=[datetime(2018, 10, 7), datetime(2020, 1, 24), datetime(2020, 2, 10)],
)


@pytest.mark.parametrize("date_pipeline", [True, False])
@pytest.mark.parametrize("example", [cons, cons_town, cons_town_doc])
def test_cons_dates(date_pipeline, example, blank_nlp):

    blank_nlp.add_pipe(
        "normalizer",
        config=dict(lowercase=True, accents=True, quotes=True, pollution=False),
    )

    if date_pipeline:
        blank_nlp.add_pipe("dates")

    blank_nlp.add_pipe(
        "consultation_dates", config=dict(**example["additionnal_params"])
    )

    doc = blank_nlp(TEXT)
    assert [
        date._.consultation_date for date in doc.spans["consultation_dates"]
    ] == example["result"]
