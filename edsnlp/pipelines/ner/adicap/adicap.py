"""`eds.adicap` pipeline"""


from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.utils.filter import filter_spans

from . import patterns
from .decoder import AdicapDecoder


class Adicap(ContextualMatcher):
    def __init__(self, nlp, pattern, attr, prefix, window):

        self.nlp = nlp
        if pattern is None:
            pattern = patterns.base_code

        if prefix is None:
            prefix = patterns.adicap_prefix

        adicap_pattern = dict(
            source="adicap",
            regex=prefix,
            regex_attr=attr,
            assign=[
                dict(
                    name="code",
                    regex=pattern,
                    window=window,
                    replace_entity=True,
                    reduce_mode=None,
                ),
            ],
        )

        super().__init__(
            nlp=nlp,
            name="adicap",
            attr=attr,
            patterns=adicap_pattern,
            ignore_excluded=False,
            regex_flags=0,
            alignment_mode="expand",
            include_assigned=False,
            assign_as_span=False,
        )

        self.decoder = AdicapDecoder()

        if not Span.has_extension("adicap"):
            Span.set_extension("adicap", default=None)

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags ADICAP mentions.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for ADICAP
        """
        spans = self.process(doc)
        spans = filter_spans(spans)

        for span in spans:
            span._.assigned = None
            span._.adicap = self.decoder.decode_adicap(span.text)

        doc.spans["adicap"] = spans

        ents, discarded = filter_spans(list(doc.ents) + spans, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
