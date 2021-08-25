from spacy.tokens import Span
from spacy import Language
from edsnlp.base import BaseComponent
from typing import Optional


class QuickUMLSComponent(BaseComponent):
    """
    This creates a QuickUMLS spaCy component which can be used in modular pipelines.

    Arguments
    ---------
    nlp:
        Existing spaCy pipeline.  This is needed to update the vocabulary with UMLS CUI values
    distribution:
        Path to QuickUMLS data
    best_match:
        Whether to return only the top match or all overlapping candidates. Defaults to True.
    ignore_syntax:
        Whether to use the heuristics introduced in the paper (Soldaini and Goharian, 2016).
    **kwargs:
        QuickUMLS keyword arguments (see QuickUMLS in core.py)
    """

    def __init__(
        self,
        nlp: Language,
        distribution: str,
        best_match: Optional[bool] = True,
        ignore_syntax: Optional[bool] = False,
        **kwargs
    ):

        from quickumls import QuickUMLS

        self.quickumls = QuickUMLS(
            distribution,
            spacy_component=True,
            **kwargs,
        )

        # save this off so that we can get vocab values of labels later
        self.nlp = nlp

        # keep these for matching
        self.best_match = best_match
        self.ignore_syntax = ignore_syntax

        # let's extend this with some properties that we want
        if not Span.has_extension("similarity"):
            Span.set_extension("similarity", default=-1.0)
            Span.set_extension("semtypes", default=-1.0)

    def __call__(self, doc):
        # pass in the document which has been parsed to this point in the pipeline for ngrams and matches
        matches = self.quickumls._match(
            doc, best_match=self.best_match, ignore_syntax=self.ignore_syntax
        )
        ents = []

        # Convert QuickUMLS match objects into Spans
        for match in matches:
            # each match may match multiple ngrams
            for ngram_match_dict in match:
                start_char_idx = int(ngram_match_dict["start"])
                end_char_idx = int(ngram_match_dict["end"])

                cui = ngram_match_dict["cui"]
                # add the string to the spacy vocab
                self.nlp.vocab.strings.add(cui)
                # pull out the value
                cui_label_value = self.nlp.vocab.strings[cui]

                # char_span() creates a Span from these character indices
                # UMLS CUI should work well as the label here
                span = doc.char_span(
                    start_char_idx, end_char_idx, label=cui_label_value
                )
                # add some custom metadata to the spans
                span._.similarity = ngram_match_dict["similarity"]
                span._.semtypes = ngram_match_dict["semtypes"]
                ents.append(span)

        doc.ents = self._filter_matches(ents)

        return doc
