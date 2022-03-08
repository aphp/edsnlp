from typing import Any, Callable, List, Union

from spacy import registry
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.advanced import AdvancedRegex
from edsnlp.utils.filter import filter_spans


class Score(AdvancedRegex):
    """
    Matcher component to extract a numeric score

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    score_name : str
        The name of the extracted score
    regex : List[str]
        A list of regexes to identify the score
    attr : str
        Wether to match on the text ('TEXT') or on the normalized text ('NORM')
    after_extract : str
        Regex with capturing group to get the score value
    score_normalization : Callable[[Union[str,None]], Any]
        Function that takes the "raw" value extracted from the `after_extract` regex,
        and should return
        - None if no score could be extracted
        - The desired score value else
    window : int
        Number of token to include after the score's mention to find the
        score's value
    """

    def __init__(
        self,
        nlp: Language,
        score_name: str,
        regex: List[str],
        attr: str,
        after_extract: str,
        score_normalization: Union[str, Callable[[Union[str, None]], Any]],
        window: int,
        verbose: int,
        ignore_excluded: bool,
    ):

        regex_config = {
            score_name: dict(regex=regex, attr=attr, after_extract=after_extract)
        }

        super().__init__(
            nlp=nlp,
            regex_config=regex_config,
            window=window,
            verbose=verbose,
            ignore_excluded=ignore_excluded,
            attr=attr,
        )

        self.score_name = score_name

        if isinstance(score_normalization, str):
            self.score_normalization = registry.get("misc", score_normalization)
        else:
            self.score_normalization = score_normalization

        self.set_extensions()

    @staticmethod
    def set_extensions() -> None:
        super(Score, Score).set_extensions()
        if not Span.has_extension("score_name"):
            Span.set_extension("score_name", default=None)
        if not Span.has_extension("score_value"):
            Span.set_extension("score_value", default=None)

    def __call__(self, doc: Doc) -> Doc:
        """
        Adds spans to document.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for extracted terms.
        """

        ents = super(Score, Score).process(self, doc)
        ents = self.score_filtering(ents)

        ents, discarded = filter_spans(list(doc.ents) + ents, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc

    def score_filtering(self, ents: List[Span]) -> List[Span]:
        """
        Extracts, if available, the value of the score.
        Normalizes the score via the provided `self.score_normalization` method.

        Parameters
        ----------
        ents: List[Span]
            List of spaCy's spans extracted by the score matcher

        Returns
        -------
        ents: List[Span]
            List of spaCy's spans, with, if found, an added `score_value` extension
        """
        to_keep_ents = []
        for ent in ents:
            value = ent._.after_extract[0]
            normalized_value = self.score_normalization(value)
            if normalized_value is not None:
                ent._.score_name = self.score_name
                ent._.score_value = int(value)
                to_keep_ents.append(ent)

        return to_keep_ents
