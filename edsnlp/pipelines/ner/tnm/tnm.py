"""`eds.tnm` pipeline."""
from typing import Dict, List, Optional, Tuple, Union

from pydantic import ValidationError
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseNERComponent, SpanSetterArg
from edsnlp.utils.filter import filter_spans

from .model import TNM
from .patterns import tnm_pattern


class TNMMatcher(BaseNERComponent):
    """
    The `eds.tnm` component extracts [TNM](https://enwp.org/wiki/TNM_staging_system)
    mentions from clinical documents.

    Examples
    --------
    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.tnm")

    text = "TNM: pTx N1 M1"

    doc = nlp(text)
    doc.ents
    # Out: (pTx N1 M1,)

    ent = doc.ents[0]
    ent._.tnm.dict()
    # {'modifier': 'p',
    #  'tumour': None,
    #  'tumour_specification': 'x',
    #  'node': '1',
    #  'node_specification': None,
    #  'metastasis': '1',
    #  'resection_completeness': None,
    #  'version': None,
    #  'version_year': None}
    ```

    Parameters
    ----------
    nlp : Optional[Language]
        The pipeline object
    name : str
        The name of the pipe
    pattern : Optional[Union[List[str], str]]
        The regex pattern to use for matching ADICAP codes
    attr : str
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    label : str
        Label name to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Authors and citation
    --------------------
    The TNM score is based on the development of S. Priou, B. Rance and
    E. Kempf ([@kempf:hal-03519085]).
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Optional[Language],
        name: str = "eds.tnm",
        *,
        pattern: Optional[Union[List[str], str]] = tnm_pattern,
        attr: str = "TEXT",
        label: str = "tnm",
        span_setter: SpanSetterArg = {"ents": True, "tnm": True},
    ):
        self.label = label

        super().__init__(nlp=nlp, name=name, span_setter=span_setter)

        if isinstance(pattern, str):
            pattern = [pattern]

        self.regex_matcher = RegexMatcher(attr=attr, alignment_mode="strict")
        self.regex_matcher.add(self.label, pattern)

    def set_extensions(self) -> None:
        """
        Set spaCy extensions
        """
        super().set_extensions()

        if not Span.has_extension(self.label):
            Span.set_extension(self.label, default=None)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find TNM mentions in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        spans:
            list of tnm spans
        """

        spans = self.regex_matcher(
            doc,
            as_spans=True,
            return_groupdict=True,
        )

        spans = filter_spans(spans)

        return spans

    def parse(self, spans: List[Tuple[Span, Dict[str, str]]]) -> List[Span]:
        """
        Parse dates using the groupdict returned by the matcher.

        Parameters
        ----------
        spans : List[Tuple[Span, Dict[str, str]]]
            List of tuples containing the spans and groupdict
            returned by the matcher.

        Returns
        -------
        List[Span]
            List of processed spans, with the date parsed.
        """

        for span, groupdict in spans:
            try:
                value = TNM.parse_obj(groupdict)
            except ValidationError:
                value = TNM.parse_obj({})

            span._.set(self.label, value)
            span.kb_id_ = value.norm()

        return [span for span, _ in spans]

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags TNM mentions.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for TNM
        """
        spans = self.process(doc)
        spans = self.parse(spans)
        self.set_spans(doc, spans)
        return doc
