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
            regex=pattern,
            regex_attr=attr,
            assign=[
                dict(
                    name="type_code",
                    regex=prefix,
                    window=window,
                    expand_entity=False,
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
            alignment_mode="strict",
            assign_as_span=True,
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

        valid_spans = []
        for span in spans:
            if span._.assigned:
                valid_spans.append(span)
                span._.assigned = None
                span._.adicap = self.decoder.decode_adicap(span.text)

        doc.spans["adicap"] = valid_spans

        ents, discarded = filter_spans(
            list(doc.ents) + valid_spans, return_discarded=True
        )

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
