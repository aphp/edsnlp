import re
from typing import Any, Callable, Dict, List, Union

from spacy.language import Language
from spacy.tokens import Span

from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.ner.scores import Score


class Sofa(Score):
    """
    Matcher component to extract the SOFA score

    Parameters
    ----------
    nlp : Language
        The spaCy object.
    score_name : str
        The name of the extracted score
    regex : List[str]
        A list of regexes to identify the SOFA score
    attr : str
        Wether to match on the text ('TEXT') or on the normalized text ('CUSTOM_NORM')
    method_regex : str
        Regex with capturing group to get the score extraction method
        (e.g. "à l'admission", "à 24H", "Maximum")
    value_regex : str
        Regex to extract the score value
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
        after_extract: List[Dict[str, str]],
        score_normalization: Union[str, Callable[[Union[str, None]], Any]],
        window: int,
        flags: Union[re.RegexFlag, int],
        ignore_excluded: bool,
    ):

        super().__init__(
            nlp,
            score_name=score_name,
            regex=regex,
            after_extract=after_extract,
            score_normalization=score_normalization,
            attr=attr,
            window=window,
            flags=flags,
            ignore_excluded=ignore_excluded,
        )

        self.set_extensions()

    @staticmethod
    def set_extensions() -> None:
        super(Sofa, Sofa).set_extensions()
        if not Span.has_extension("score_method"):
            Span.set_extension("score_method", default=None)

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
            assigned = ent._.assigned
            if not assigned:
                continue
            if assigned.get("method_max") is not None:
                method = "Maximum"
                value = assigned["method_max"]
            elif assigned.get("method_24h") is not None:
                method = "24H"
                value = assigned["method_24h"]
            elif assigned.get("method_adm") is not None:
                method = "A l'admission"
                value = assigned["method_adm"]
            else:
                method = "Non précisée"
                value = assigned["no_method"]
            text_value = get_text(
                value,
                self.attr,
                self.ignore_excluded,
            )
            normalized_value = self.score_normalization(text_value)

            if normalized_value is not None:
                value._.score_name = self.score_name
                value._.score_value = int(normalized_value)
                value._.score_method = method
                to_keep_ents.append(value)

        return to_keep_ents
