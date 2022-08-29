from typing import List, Union

from spacy.language import Language

from edsnlp.pipelines.misc.consultation_dates import ConsultationDates
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    consultation_mention=True,
    town_mention=False,
    document_date_mention=False,
    attr="NORM",
)


@deprecated_factory(
    "consultation_dates",
    "eds.consultation_dates",
    default_config=DEFAULT_CONFIG,
    assigns=["doc._.consultation_dates"],
)
@Language.factory(
    "eds.consultation_dates",
    default_config=DEFAULT_CONFIG,
    assigns=["doc._.consultation_dates"],
)
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
