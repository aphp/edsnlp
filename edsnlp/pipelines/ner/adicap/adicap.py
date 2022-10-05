"""`eds.adicap` pipeline"""


from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.utils.filter import filter_spans
from edsnlp.utils.resources import get_adicap_dict

from . import patterns
from .models import AdicapCode


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

        self.decode_dict = get_adicap_dict()

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:
        super().set_extensions()
        if not Span.has_extension("adicap"):
            Span.set_extension("adicap", default=None)
        if not Span.has_extension("value"):
            Span.set_extension("value", default=None)

    def decode(self, code):
        exploded = list(code)
        adicap = AdicapCode(
            code=code,
            sampling_mode=self.decode_dict["D1"]["codes"].get(exploded[0]),
            technic=self.decode_dict["D2"]["codes"].get(exploded[1]),
            organ=self.decode_dict["D3"]["codes"].get("".join(exploded[2:4])),
        )

        for d in ["D4", "D5", "D6", "D7"]:
            adicap_short = self.decode_dict[d]["codes"].get("".join(exploded[4:8]))
            adicap_long = self.decode_dict[d]["codes"].get("".join(exploded[2:8]))

            if (adicap_short is not None) | (adicap_long is not None):
                adicap.pathology = self.decode_dict[d]["label"]
                adicap.behaviour_type = self.decode_dict[d]["codes"].get(exploded[5])

                if adicap_short is not None:
                    adicap.pathology_type = adicap_short

                else:
                    adicap.pathology_type = adicap_long

        return adicap

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
            span._.adicap = self.decode(span._.assigned["code"])
            span._.value = span._.adicap
            span._.assigned = None

        doc.spans["adicap"] = spans

        ents, discarded = filter_spans(list(doc.ents) + spans, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
