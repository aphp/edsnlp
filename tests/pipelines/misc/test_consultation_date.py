import pytest

from edsnlp.pipelines.misc.consultation_dates import factory  # noqa

TEXT = """
Références : AMO/AMO
Objet : Compte-Rendu de Consultation du 07/10/2018
Madame BEESLY Pamela, âgée de 45 ans, née le 05/10/1987, a été vue en consultation
dans le service de NCK CS RHUMATO. Tel: 01-02-03-04-05

####

CR CS 3-1-2019 1/2

####

Paris, le 24 janvier 2020

####

Document signé le 10/02/2020

"""

cons = dict(
    additional_params=dict(),
    result=[
        dict(year=2018, month=10, day=7),
        dict(year=2019, month=1, day=3),
    ],
)

cons_town = dict(
    additional_params=dict(town_mention=True),
    result=[
        dict(year=2018, month=10, day=7),
        dict(year=2019, month=1, day=3),
        dict(year=2020, month=1, day=24),
    ],
)

cons_town_doc = dict(
    additional_params=dict(town_mention=True, document_date_mention=True),
    result=[
        dict(year=2018, month=10, day=7),
        dict(year=2019, month=1, day=3),
        dict(year=2020, month=1, day=24),
        dict(year=2020, month=2, day=10),
    ],
)


@pytest.mark.parametrize("date_pipeline", [True, False])
@pytest.mark.parametrize("example", [cons, cons_town, cons_town_doc])
def test_cons_dates(date_pipeline, example, blank_nlp):

    blank_nlp.add_pipe(
        "eds.normalizer",
        config=dict(lowercase=True, accents=True, quotes=True, pollution=False),
    )

    blank_nlp.add_pipe(
        "eds.consultation_dates", config=dict(**example["additional_params"])
    )

    if date_pipeline:
        blank_nlp.add_pipe("eds.dates")

    doc = blank_nlp(TEXT)

    assert not date_pipeline or len(doc.spans["dates"]) == 5

    assert len(doc.spans["consultation_dates"]) == len(example["result"])

    for span, result in zip(doc.spans["consultation_dates"], example["result"]):
        assert all([span._.consultation_date.dict()[k] == result[k] for k in result])
