from typing import List, Union

from spacy.language import Language

from edsnlp.pipelines.consultation_dates import ConsultationDates

consultation_date_default_config = dict(
    consultation_mention=True,
    town_mention=False,
    document_date_mention=False,
    attr="NORM",
)


@Language.factory("consultation_dates", default_config=consultation_date_default_config)
def create_component(
    nlp: Language,
    name: str,
    attr: str,
    consultation_mention: Union[List[str], bool],
    town_mention: Union[List[str], bool],
    document_date_mention: Union[List[str], bool],
):
    return ConsultationDates(
        nlp,
        attr=attr,
        consultation_mention=consultation_mention,
        document_date_mention=document_date_mention,
        town_mention=town_mention,
    )
