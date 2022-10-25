from datetime import timedelta
from typing import List, Optional

import pendulum
from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.qualifiers.base import Qualifier
from edsnlp.pipelines.terminations import termination
from edsnlp.utils.deprecation import deprecated_getter_factory
from edsnlp.utils.filter import consume_spans, filter_spans, get_spans
from edsnlp.utils.inclusion import check_inclusion, check_sent_inclusion

from .patterns import history, sections_history


class History(Qualifier):
    """
    Implements an history detection algorithm.

    The components looks for terms indicating history in the text.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    history : Optional[List[str]]
        List of terms indicating medical history reference.
    termination : Optional[List[str]]
        List of syntagme termination terms.
    use_sections : bool
        Whether to use section pipeline to detect medical history section.
    use_dates : bool
        Whether to use dates pipeline to detect if the event occurs
         a long time before the document date.
    history_limit : int
        The number of days after which the event is considered as history.
    exclude_birthdate : bool
        Whether to exclude the birth date from history dates.
    closest_dates_only : bool
        Whether to include the closest dates only.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    on_ents_only : bool
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    regex : Optional[Dict[str, Union[List[str], str]]]
        A dictionary of regex patterns.
    explain : bool
        Whether to keep track of cues for each entity.
    """

    defaults = dict(
        history=history,
        termination=termination,
    )

    def __init__(
        self,
        nlp: Language,
        attr: str,
        history: Optional[List[str]],
        termination: Optional[List[str]],
        use_sections: bool,
        use_dates: bool,
        history_limit: int,
        closest_dates_only: bool,
        exclude_birthdate: bool,
        explain: bool,
        on_ents_only: bool,
    ):

        terms = self.get_defaults(
            history=history,
            termination=termination,
        )

        super().__init__(
            nlp=nlp,
            attr=attr,
            on_ents_only=on_ents_only,
            explain=explain,
            **terms,
        )

        self.set_extensions()

        self.history_limit = timedelta(history_limit)
        self.exclude_birthdate = exclude_birthdate
        self.closest_dates_only = closest_dates_only

        self.sections = use_sections and (
            "eds.sections" in nlp.pipe_names or "sections" in nlp.pipe_names
        )
        if use_sections and not self.sections:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `section` pipeline, but it was not set. "
                "Skipping that step."
            )

        self.dates = use_dates and (
            "eds.dates" in nlp.pipe_names or "dates" in nlp.pipe_names
        )
        if use_dates:
            if not self.dates:
                logger.warning(
                    "You have requested that the pipeline use dates "
                    "provided by the `dates` pipeline, but it was not set. "
                    "Skipping that step."
                )
            elif exclude_birthdate:
                logger.info(
                    "You have requested that the pipeline use date "
                    "and exclude birth dates. "
                    "To make the most of this feature, "
                    "make sur you provide the `birth_datetime` "
                    "context and `note_datetime` context. "
                )
            else:
                logger.info(
                    "You have requested that the pipeline use date "
                    "To make the most of this feature, "
                    "make sure you provide the `note_datetime` "
                    "context. "
                )

    @classmethod
    def set_extensions(cls) -> None:

        if not Token.has_extension("history"):
            Token.set_extension("history", default=False)

        if not Token.has_extension("antecedents"):
            Token.set_extension(
                "antecedents",
                getter=deprecated_getter_factory("antecedents", "history"),
            )

        if not Token.has_extension("antecedent"):
            Token.set_extension(
                "antecedent",
                getter=deprecated_getter_factory("antecedent", "history"),
            )

        if not Token.has_extension("history_"):
            Token.set_extension(
                "history_",
                getter=lambda token: "ATCD" if token._.history else "CURRENT",
            )

        if not Token.has_extension("antecedents_"):
            Token.set_extension(
                "antecedents_",
                getter=deprecated_getter_factory("antecedents_", "history_"),
            )

        if not Token.has_extension("antecedent_"):
            Token.set_extension(
                "antecedent_",
                getter=deprecated_getter_factory("antecedent_", "history_"),
            )

        if not Span.has_extension("history"):
            Span.set_extension("history", default=False)

        if not Span.has_extension("antecedents"):
            Span.set_extension(
                "antecedents",
                getter=deprecated_getter_factory("antecedents", "history"),
            )

        if not Span.has_extension("antecedent"):
            Span.set_extension(
                "antecedent",
                getter=deprecated_getter_factory("antecedent", "history"),
            )

        if not Span.has_extension("history_"):
            Span.set_extension(
                "history_",
                getter=lambda span: "ATCD" if span._.history else "CURRENT",
            )

        if not Span.has_extension("antecedents_"):
            Span.set_extension(
                "antecedents_",
                getter=deprecated_getter_factory("antecedents_", "history_"),
            )

        if not Span.has_extension("antecedent_"):
            Span.set_extension(
                "antecedent_",
                getter=deprecated_getter_factory("antecedent_", "history_"),
            )
        # Store history mentions responsible for the history entity's character
        if not Span.has_extension("history_cues"):
            Span.set_extension("history_cues", default=[])

        # Store recent mentions responsible for the non-antecedent entity's character
        if not Span.has_extension("recent_cues"):
            Span.set_extension("recent_cues", default=[])

        if not Span.has_extension("antecedents_cues"):
            Span.set_extension(
                "antecedents_cues",
                getter=deprecated_getter_factory("antecedents_cues", "history_cues"),
            )

        if not Span.has_extension("antecedent_cues"):
            Span.set_extension(
                "antecedent_cues",
                getter=deprecated_getter_factory("antecedent_cues", "history_cues"),
            )

    def process(self, doc: Doc) -> Doc:
        """
        Finds entities related to history.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for history
        """

        if doc._.note_datetime is not None:
            try:
                note_datetime = pendulum.instance(doc._.note_datetime)
                note_datetime = note_datetime.set(tz="Europe/Paris")
            except ValueError:
                logger.debug(
                    "note_datetime must be a datetime objects. "
                    "Skipping history qualification from note_datetime."
                )
                note_datetime = None

        if doc._.birth_datetime is not None:
            try:
                birth_datetime = pendulum.instance(doc._.birth_datetime)
                birth_datetime = birth_datetime.set(tz="Europe/Paris")
            except ValueError:
                logger.debug(
                    "birth_datetime must be a datetime objects. "
                    "Skipping history qualification from birth date."
                )
                birth_datetime = None

        matches = self.get_matches(doc)

        terminations = get_spans(matches, "termination")
        boundaries = self._boundaries(doc, terminations)

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches, label_to_remove="pseudo")

        entities = list(doc.ents) + list(doc.spans.get("discarded", []))
        ents = None
        sub_sections = None
        sub_recent_dates = None
        sub_history_dates = None

        sections = []
        if self.sections:
            sections = [
                Span(doc, section.start, section.end, label="ATCD")
                for section in doc.spans["sections"]
                if section.label_ in sections_history
            ]

        history_dates = []
        recent_dates = []
        if self.dates:
            for date in doc.spans["dates"]:
                if date.label_ == "relative":
                    if date._.date.direction.value == "CURRENT":
                        if (
                            (
                                date._.date.year == 0
                                and self.history_limit >= timedelta(365)
                            )
                            or (
                                date._.date.month == 0
                                and self.history_limit >= timedelta(30)
                            )
                            or (
                                date._.date.week == 0
                                and self.history_limit >= timedelta(7)
                            )
                            or (date._.date.day == 0)
                        ):
                            recent_dates.append(
                                Span(doc, date.start, date.end, label="relative_date")
                            )
                    elif date._.date.direction.value == "PAST":
                        if -date._.date.to_datetime() >= self.history_limit:
                            history_dates.append(
                                Span(doc, date.start, date.end, label="relative_date")
                            )
                        else:
                            recent_dates.append(
                                Span(doc, date.start, date.end, label="relative_date")
                            )
                elif date.label_ == "absolute" and doc._.note_datetime:
                    try:
                        absolute_date = date._.date.to_datetime(
                            note_datetime=note_datetime,
                            infer_from_context=True,
                            tz="Europe/Paris",
                            default_day=15,
                        )
                    except ValueError as e:
                        absolute_date = None
                        logger.warning(
                            "In doc {}, the following date {} raises this error: {}. "
                            "Skipping this date.",
                            doc._.note_id,
                            date._.date,
                            e,
                        )
                    if absolute_date:
                        if note_datetime.diff(absolute_date) < self.history_limit:
                            recent_dates.append(
                                Span(doc, date.start, date.end, label="absolute_date")
                            )
                        elif not (
                            self.exclude_birthdate
                            and birth_datetime
                            and absolute_date == birth_datetime
                        ):
                            history_dates.append(
                                Span(doc, date.start, date.end, label="absolute_date")
                            )

        for start, end in boundaries:
            ents, entities = consume_spans(
                entities,
                filter=lambda s: check_inclusion(s, start, end),
                second_chance=ents,
            )

            sub_matches, matches = consume_spans(
                matches, lambda s: start <= s.start < end
            )

            if self.sections:
                sub_sections, sections = consume_spans(
                    sections, lambda s: s.start < end <= s.end, sub_sections
                )
            if self.dates:
                sub_recent_dates, recent_dates = consume_spans(
                    recent_dates,
                    lambda s: check_sent_inclusion(s, start, end),
                    sub_recent_dates,
                )
                sub_history_dates, history_dates = consume_spans(
                    history_dates,
                    lambda s: check_sent_inclusion(s, start, end),
                    sub_history_dates,
                )

                # Filter dates inside the boundaries only
                if self.closest_dates_only:
                    close_recent_dates = []
                    close_history_dates = []
                    if sub_recent_dates:
                        close_recent_dates = [
                            recent_date
                            for recent_date in sub_recent_dates
                            if check_inclusion(recent_date, start, end)
                        ]
                        if sub_history_dates:
                            close_history_dates = [
                                history_date
                                for history_date in sub_history_dates
                                if check_inclusion(history_date, start, end)
                            ]
                            # If no date inside the boundaries, get the closest
                            if not close_recent_dates and not close_history_dates:
                                min_distance_recent_date = min(
                                    [
                                        abs(sub_recent_date.start - start)
                                        for sub_recent_date in sub_recent_dates
                                    ]
                                )
                                min_distance_history_date = min(
                                    [
                                        abs(sub_history_date.start - start)
                                        for sub_history_date in sub_history_dates
                                    ]
                                )
                                if min_distance_recent_date < min_distance_history_date:
                                    close_recent_dates = [
                                        min(
                                            sub_recent_dates,
                                            key=lambda x: abs(x.start - start),
                                        )
                                    ]
                                else:
                                    close_history_dates = [
                                        min(
                                            sub_history_dates,
                                            key=lambda x: abs(x.start - start),
                                        )
                                    ]
                        elif not close_recent_dates:
                            close_recent_dates = [
                                min(
                                    sub_recent_dates,
                                    key=lambda x: abs(x.start - start),
                                )
                            ]
                    elif sub_history_dates:
                        close_history_dates = [
                            history_date
                            for history_date in sub_history_dates
                            if check_inclusion(history_date, start, end)
                        ]
                        # If no date inside the boundaries, get the closest
                        if not close_history_dates:
                            close_history_dates = [
                                min(
                                    sub_history_dates,
                                    key=lambda x: abs(x.start - start),
                                )
                            ]

            if self.on_ents_only and not ents:
                continue

            history_cues = get_spans(sub_matches, "history")
            recent_cues = []

            if self.sections:
                history_cues.extend(sub_sections)

            if self.dates:
                history_cues.extend(
                    close_history_dates
                    if self.closest_dates_only
                    else sub_history_dates
                )
                recent_cues.extend(
                    close_recent_dates if self.closest_dates_only else sub_recent_dates
                )

            history = bool(history_cues) and not bool(recent_cues)

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.history = history

            for ent in ents:
                ent._.history = ent._.history or history

                if self.explain:
                    ent._.history_cues += history_cues
                    ent._.recent_cues += recent_cues

                if not self.on_ents_only and ent._.history:
                    for token in ent:
                        token._.history = True

        return doc
