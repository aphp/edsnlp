from edsnlp.core import registry

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

create_component = registry.factory.register(
    "eds.consultation_dates",
    assigns=["doc.spans", "doc.ents"],
    deprecated=["consultation_dates"],
)(ConsultationDatesMatcher)
