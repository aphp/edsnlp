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
    additional_params=dict(),
    result=[dict(year=2018, month=10, day=7)],
)

cons_town = dict(
    additional_params=dict(town_mention=True),
    result=[dict(year=2018, month=10, day=7), dict(year=2020, month=1, day=24)],
)

cons_town_doc = dict(
    additional_params=dict(town_mention=True, document_date_mention=True),
    result=[
        dict(year=2018, month=10, day=7),
        dict(year=2020, month=1, day=24),
        dict(year=2020, month=2, day=10),
    ],
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
        "consultation_dates", config=dict(**example["additional_params"])
    )

    doc = blank_nlp(TEXT)

    assert len(doc.spans["dates"]) == 4 or not date_pipeline

    assert len(doc.spans["consultation_dates"]) == len(example["result"])

    for span, result in zip(doc.spans["consultation_dates"], example["result"]):
        assert all([span._.consultation_date.dict()[k] == result[k] for k in result])
