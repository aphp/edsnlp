from typing import List, Optional, Set, Union

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import Dates

DEFAULT_CONFIG = dict(
    absolute=None,
    relative=None,
    duration=None,
    false_positive=None,
    detect_periods=False,
    detect_time=True,
    on_ents_only=False,
    as_ents=False,
    attr="LOWER",
)


@deprecated_factory(
    "dates", "eds.dates", default_config=DEFAULT_CONFIG, assigns=["doc.spans"]
)
@Language.factory("eds.dates", default_config=DEFAULT_CONFIG, assigns=["doc.spans"])
def create_component(
    nlp: Language,
    name: str = "eds.dates",
    absolute: Optional[List[str]] = None,
    relative: Optional[List[str]] = None,
    duration: Optional[List[str]] = None,
    false_positive: Optional[List[str]] = None,
    on_ents_only: Union[bool, str, List[str], Set[str]] = False,
    detect_periods: bool = False,
    detect_time: bool = True,
    as_ents: bool = False,
    attr: str = "LOWER",
):
    """
    Tags and normalizes dates, using the open-source `dateparser` library.

    The pipeline uses spaCy's `filter_spans` function.
    It filters out false positives, and introduce a hierarchy between patterns.
    For instance, in case of ambiguity, the pipeline will decide that a date is a
    date without a year rather than a date without a day.

    Parameters
    ----------
    nlp : spacy.language.Language
        Language pipeline object
    absolute : Union[List[str], str]
        List of regular expressions for absolute dates.
    relative : Union[List[str], str]
        List of regular expressions for relative dates
        (eg `hier`, `la semaine prochaine`).
    duration : Union[List[str], str]
        List of regular expressions for durations
        (eg `pendant trois mois`).
    false_positive : Union[List[str], str]
        List of regular expressions for false positive (eg phone numbers, etc).
    on_ents_only : Union[bool, str, List[str]]
        Whether to look on dates in the whole document or in specific sentences:

        - If `True`: Only look in the sentences of each entity in doc.ents
        - If False: Look in the whole document
        - If given a string `key` or list of string: Only look in the sentences of
          each entity in `#!python doc.spans[key]`
    detect_periods : bool
        Whether to detect periods (experimental)
    detect_time: bool
        Whether to detect time inside dates
    as_ents : bool
        Whether to treat dates as entities
    attr : str
        spaCy attribute to use
    """
    return Dates(
        nlp,
        name=name,
        absolute=absolute,
        relative=relative,
        duration=duration,
        false_positive=false_positive,
        on_ents_only=on_ents_only,
        detect_periods=detect_periods,
        detect_time=detect_time,
        as_ents=as_ents,
        attr=attr,
    )
