from spacy.tokens import Doc

from edsnlp.core import PipelineProtocol, registry


def remove_lowercase(doc: Doc):
    """
    Add case on the `NORM` custom attribute. Should always be applied first.

    Parameters
    ----------
    doc : Doc
        The spaCy `Doc` object.

    Returns
    -------
    Doc
        The document, with case put back in `NORM`.
    """

    for token in doc:
        token.norm_ = token.text

    return doc


@registry.factory.register(
    "eds.remove_lowercase",
    assigns=["token.norm"],
    deprecated=[
        "remove-lowercase",
        "eds.remove-lowercase",
    ],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
):
    """
    Add case on the `NORM` custom attribute. Should always be applied first.

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object.
    name : str
        The name of the component.
    """
    return remove_lowercase  # pragma: no cover
