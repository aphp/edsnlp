from spacy import registry

from edsnlp import components  # noqa : used to update the registry


def get_all_pipes():
    """
    Get all comorbidity pipe names

    Returns
    -------
    _type_
        _description_
    """
    return [
        name
        for name in registry.factories.get_all().keys()
        if ".comorbidities." in name
    ]
