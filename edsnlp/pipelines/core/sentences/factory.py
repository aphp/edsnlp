from typing import List, Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import SentenceSegmenter

DEFAULT_CONFIG = dict(
    punct_chars=None,
    ignore_excluded=True,
    use_endlines=None,
)


@deprecated_factory(
    "sentences",
    "eds.sentences",
    assigns=["token.is_sent_start"],
)
@Language.factory(
    "eds.sentences",
    assigns=["token.is_sent_start"],
)
def create_component(
    nlp: Language,
    name: str = "eds.sentences",
    *,
    punct_chars: Optional[List[str]] = None,
    use_endlines: Optional[bool] = True,
    ignore_excluded: bool = None,
):
    r'''
    The `eds.sentences` matcher provides an alternative to spaCy's default
    `sentencizer`, aiming to overcome some of its limitations.

    Indeed, the `sentencizer` merely looks at period characters to detect the end of a
    sentence, a strategy that often fails in a clinical note settings. Our
    `eds.sentences` component also classifies end-of-lines as sentence boundaries if
    the subsequent token begins with an uppercase character, leading to slightly better
    performances.

    Moreover, the `eds.sentences` component use the output of the `eds.normalizer`
    and `eds.endlines` output by default when these components are added to the
    pipeline.

    Examples
    --------
    === "EDS-NLP"

        ```{ .python .no-check }
        import spacy

        nlp = spacy.blank("eds")
        nlp.add_pipe("eds.sentences")

        text = """Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        Il lui était arrivé la même chose il y a deux ans."
        """

        doc = nlp(text)

        for sentence in doc.sents:
            print("<s>", sentence, "</s>")
        # Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        # Out:  <\s>
        # Out: <s> Il lui était arrivé la même chose il y a deux ans. <\s>
        ```

    === "spaCy sentencizer"

        ```{ .python .no-check }
        import spacy

        nlp = spacy.blank("eds")
        nlp.add_pipe("sentencizer")

        text = """Le patient est admis le 23 août 2021 pour une douleur à l'estomac"
        Il lui était arrivé la même chose il y a deux ans.
        """

        doc = nlp(text)

        for sentence in doc.sents:
            print("<s>", sentence, "</s>")
        # Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        # Out: Il lui était arrivé la même chose il y a deux ans. <\s>
        ```

    Notice how EDS-NLP's implementation is more robust to ill-defined sentence endings.


    Parameters
    ----------
    nlp: Language
        The pipeline object.
    name: str
        The name of the component.
    punct_chars : Optional[List[str]]
        Punctuation characters.
    use_endlines : bool
        Whether to use endlines prediction.
    ignore_excluded : bool
        Whether to skip excluded tokens (requires the upstream `eds.normalizer` pipe).

    Authors and citation
    --------------------
    The `eds.sentences` component was developed by AP-HP's Data Science team.
    '''

    return SentenceSegmenter(
        nlp=nlp,
        name=name,
        punct_chars=punct_chars,
        use_endlines=use_endlines,
        ignore_excluded=ignore_excluded,
    )
