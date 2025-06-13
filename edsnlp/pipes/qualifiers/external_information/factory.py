from edsnlp.core import registry
from edsnlp.pipes.qualifiers.external_information.external_information import (
    ExternalInformationQualifier,
)

create_component = registry.factory.register(
    "eds.external_information_qualifier",
    assigns=["doc.spans"],
)(ExternalInformationQualifier)
