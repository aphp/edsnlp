import warnings
from typing import List, Optional

from spacy.tokens import Doc

from edsnlp.core import PipelineProtocol

from ...base import BaseComponent
from .fast_sentences import FastSentenceSegmenter
from .terms import punctuation


class SentenceSegmenter(BaseComponent):
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
        import edsnlp, edsnlp.pipes as eds

        nlp = edsnlp.blank("eds")
        nlp.add_pipe(eds.sentences())  # same as nlp.add_pipe("eds.sentences")

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
        import edsnlp, edsnlp.pipes as eds

        nlp = edsnlp.blank("eds")
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
    nlp: PipelineProtocol
        The EDS-NLP pipeline
    name: Optional[str]
        The name of the component
    punct_chars: Optional[List[str]]
        Punctuation characters.
    use_endlines: bool
        Whether to use endlines prediction.
    ignore_excluded: bool
        Whether to ignore excluded tokens.
    check_capitalized: bool
        Whether to check for capitalized words after newlines or full stops.
    min_newline_count: int
        The minimum number of newlines to consider a newline-triggered sentence.

    Authors and citation
    --------------------
    The `eds.sentences` component was developed by AP-HP's Data Science team.
    '''

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: Optional[str] = "sentences",
        punct_chars: Optional[List[str]] = None,
        use_endlines: Optional[bool] = None,
        ignore_excluded: bool = True,
        check_capitalized: bool = True,
        min_newline_count: int = 1,
    ):
        super().__init__(nlp, name)
        if min_newline_count > 1 and nlp.lang != "eds":
            warnings.warn(
                "To use min_newline_count > 1, you need to use the 'eds' language "
                "in order to split newlines into separate and countable tokens."
            )

        if punct_chars is None:
            punct_chars = punctuation

        self.fast_segmenter = FastSentenceSegmenter(
            vocab=nlp.vocab,
            punct_chars=punct_chars,
            use_endlines=use_endlines,
            ignore_excluded=ignore_excluded,
            check_capitalized=check_capitalized,
            min_newline_count=min_newline_count,
        )

    def __call__(self, doc: Doc):
        return self.fast_segmenter(doc)
