from spacy import registry
from edsnlp import components  # noqa : used to update the registry

from spacy.tokens import Span


def get_all_pipes():
    return [
        name
        for name in registry.factories.get_all().keys()
        if ".comorbidities." in name
    ]
