from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .consultation_dates import ConsultationDatesMatcher

DEFAULT_CONFIG = dict(
    consultation_mention=True,
    town_mention=False,
    document_date_mention=False,
    attr="NORM",
    ignore_excluded=False,
    ignore_spacy_tokens=False,
    label="consultation_date",
    span_setter={"ents": True, "consultation_dates": True},
)

create_component = deprecated_factory(
    "consultation_dates",
    "eds.consultation_dates",
    assigns=["doc.spans", "doc.ents"],
)(ConsultationDatesMatcher)
create_component = Language.factory(
    "eds.consultation_dates",
    assigns=["doc.spans", "doc.ents"],
)(create_component)
