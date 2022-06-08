from typing import List, Optional, Set, Union

from spacy.language import Language

from edsnlp.pipelines.qualifiers.history import History, patterns
from edsnlp.pipelines.terminations import termination as termination_patterns
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    history=patterns.history,
    termination=termination_patterns,
    use_sections=False,
    use_dates=False,
    history_limit=14,
    exclude_birthdate=True,
    closest_dates_only=True,
    explain=False,
    on_ents_only=True,
)


@deprecated_factory(
    "antecedents",
    "eds.history",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.history"],
)
@deprecated_factory(
    "eds.antecedents",
    "eds.history",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.history"],
)
@deprecated_factory(
    "history",
    "eds.history",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.history"],
)
@Language.factory(
    "eds.history",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.history"],
)
def create_component(
    nlp: Language,
    name: str = "eds.history",
    history: Optional[List[str]] = patterns.history,
    termination: Optional[List[str]] = termination_patterns,
    use_sections: bool = False,
    use_dates: bool = False,
    history_limit: int = 14,
    exclude_birthdate: bool = True,
    closest_dates_only: bool = True,
    attr: str = "NORM",
    explain: bool = False,
    on_ents_only: Union[bool, str, List[str], Set[str]] = True,
):
    """
    Implements a history detection algorithm.

    The component looks for terms indicating history in the text.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    name : str
        Name of the component.
    history : Optional[List[str]]
        List of terms indicating medical history reference.
    termination : Optional[List[str]]
        List of syntagms termination terms.
    use_sections : bool
        Whether to use section pipeline to detect medical history section.
    use_dates : bool
        Whether to use dates pipeline to detect if the event occurs
         a long time before the document date.
    history_limit : int
        The number of days after which the event is considered as history.
    exclude_birthdate : bool
        Whether to exclude the birthdate from history dates.
    closest_dates_only : bool
        Whether to include the closest dates only.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    on_ents_only : bool
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    explain : bool
        Whether to keep track of cues for each entity.
    """
    return History(
        nlp,
        attr=attr,
        history=history,
        termination=termination,
        use_sections=use_sections,
        use_dates=use_dates,
        history_limit=history_limit,
        exclude_birthdate=exclude_birthdate,
        closest_dates_only=closest_dates_only,
        explain=explain,
        on_ents_only=on_ents_only,
    )
