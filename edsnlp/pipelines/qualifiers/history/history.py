from datetime import timedelta
from typing import List, Optional, Set, Union

import pendulum
from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.base import SpanGetterArg, get_spans
from edsnlp.pipelines.qualifiers.base import RuleBasedQualifier
from edsnlp.pipelines.terminations import termination as default_termination
from edsnlp.utils.deprecation import deprecated_getter_factory
from edsnlp.utils.filter import consume_spans, filter_spans
from edsnlp.utils.inclusion import check_inclusion, check_sent_inclusion

from . import patterns
from .patterns import sections_history


class HistoryQualifier(RuleBasedQualifier):
    """
    The `eds.history` pipeline uses a simple rule-based algorithm to detect spans that
    describe medical history rather than the diagnostic of a given visit.

    The mere definition of a medical history is not straightforward.
    Hence, this component only tags entities that are _explicitly described as part of
    the medical history_, e.g., preceded by a synonym of "medical history".

    This component may also use the output of:

    - the [`eds.sections` component](/pipelines/misc/sections/). In that case, the
    entire `antécédent` section is tagged as a medical history.

    !!! warning "Sections"

        Be careful, the `eds.sections` component may oversize the `antécédents` section.
        Indeed, it detects *section titles* and tags the entire text between a title and
        the next as a section. Hence, should a section title goes undetected after the
        `antécédents` title, some parts of the document will erroneously be tagged as
        a medical history.

        To curb that possibility, using the output of the `eds.sections` component is
        deactivated by default.

    - the [`eds.dates` component](/pipelines/misc/dates). In that case, it will take the
      dates into account to tag extracted entities as a medical history or not.

    !!! info "Dates"

        To take the most of the `eds.dates` component, you may add the ``note_datetime``
        context (cf. [Adding context][using-eds-nlps-helper-functions]). It allows the
        component to compute the duration of absolute dates
        (e.g., le 28 août 2022/August 28, 2022). The ``birth_datetime`` context allows
        the component to exclude the birthdate from the extracted dates.

    Examples
    --------
    The following snippet matches a simple terminology, and checks whether the extracted
    entities are history or not. It is complete and can be run _as is_.

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sections")
    nlp.add_pipe("eds.dates")
    nlp.add_pipe(
        "eds.matcher",
        config=dict(terms=dict(douleur="douleur", malaise="malaises")),
    )
    nlp.add_pipe(
        "eds.history",
        config=dict(
            use_sections=True,
            use_dates=True,
        ),
    )

    text = (
        "Le patient est admis le 23 août 2021 pour une douleur au bras. "
        "Il a des antécédents de malaises."
        "ANTÉCÉDENTS : "
        "- le patient a déjà eu des malaises. "
        "- le patient a eu une douleur à la jambe il y a 10 jours"
    )

    doc = nlp(text)

    doc.ents
    # Out: (douleur, malaises, malaises, douleur)

    doc.ents[0]._.history
    # Out: False

    doc.ents[1]._.history
    # Out: True

    doc.ents[2]._.history  # (1)
    # Out: True

    doc.ents[3]._.history  # (2)
    # Out: False
    ```

    1. The entity is in the section `antécédent`.
    2. The entity is in the section `antécédent`, however the extracted `relative_date`
    refers to an event that took place within 14 days.

    Extensions
    ----------
    The `eds.history` component declares two extensions, on both `Span` and `Token`
    objects :

    1. The `history` attribute is a boolean, set to `True` if the component predicts
       that the span/token is a medical history.
    2. The `history_` property is a human-readable string, computed from the `history`
       attribute. It implements a simple getter function that outputs `CURRENT` or
       `ATCD`, depending on the value of `history`.

    Parameters
    ----------
    nlp : Language
        The pipeline object.
    name : Optional[str]
        The component name.
    history : Optional[List[str]]
        List of terms indicating medical history reference.
    termination : Optional[List[str]]
        List of syntagms termination terms.
    use_sections : bool
        Whether to use section pipeline to detect medical history section.
    use_dates : bool
        Whether to use dates pipeline to detect if the event occurs
         a long time before the document date.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    history_limit : Union[int, timedelta]
        The number of days after which the event is considered as history.
    exclude_birthdate : bool
        Whether to exclude the birthdate from history dates.
    closest_dates_only : bool
        Whether to include the closest dates only.
    span_getter : SpanGetterArg
        Which entities should be classified. By default, `doc.ents`
    on_ents_only : Union[bool, str, List[str], Set[str]]
        Deprecated, use `span_getter` instead.

        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.

        - If True, will look in all ents located in `doc.ents` only
        - If an iterable of string is passed, will additionally look in `doc.spans[key]`
        for each key in the iterable
    explain : bool
        Whether to keep track of cues for each entity.

    Authors and citation
    --------------------
    The `eds.history` component was developed by AP-HP's Data Science team.
    """

    history_limit: timedelta

    def __init__(
        self,
        nlp: Language,
        name: Optional[str] = "eds.history",
        *,
        history: Optional[List[str]] = None,
        termination: Optional[List[str]] = None,
        use_sections: bool = False,
        use_dates: bool = False,
        attr: str = "NORM",
        history_limit: int = 14,
        closest_dates_only: bool = True,
        exclude_birthdate: bool = True,
        span_getter: SpanGetterArg = None,
        on_ents_only: Union[bool, str, List[str], Set[str]] = True,
        explain: bool = False,
    ):

        terms = dict(
            history=patterns.history if history is None else history,
            termination=default_termination if termination is None else termination,
        )

        super().__init__(
            nlp=nlp,
            name=name,
            attr=attr,
            explain=explain,
            terms=terms,
            on_ents_only=on_ents_only,
            span_getter=span_getter,
        )

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

    def set_extensions(self) -> None:
        for cls in (Token, Span):
            if not cls.has_extension("history"):
                cls.set_extension("history", default=False)

            if not cls.has_extension("antecedents"):
                cls.set_extension(
                    "antecedents",
                    getter=deprecated_getter_factory("antecedents", "history"),
                )

            if not cls.has_extension("antecedent"):
                cls.set_extension(
                    "antecedent",
                    getter=deprecated_getter_factory("antecedent", "history"),
                )

            if not cls.has_extension("history_"):
                cls.set_extension(
                    "history_",
                    getter=lambda token: "ATCD" if token._.history else "CURRENT",
                )

            if not cls.has_extension("antecedents_"):
                cls.set_extension(
                    "antecedents_",
                    getter=deprecated_getter_factory("antecedents_", "history_"),
                )

            if not cls.has_extension("antecedent_"):
                cls.set_extension(
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
        note_datetime = None
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

        birth_datetime = None
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

        terminations = [m for m in matches if m.label_ == "termination"]
        boundaries = self._boundaries(doc, terminations)

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches, label_to_remove="pseudo")

        entities = list(get_spans(doc, self.span_getter))
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
                value = date._.date
                if value.mode == "relative":
                    if value.direction.value == "current":
                        if (
                            (value.year == 0 and self.history_limit >= timedelta(365))
                            or (
                                value.month == 0 and self.history_limit >= timedelta(30)
                            )
                            or (value.week == 0 and self.history_limit >= timedelta(7))
                            or (value.day == 0)
                        ):
                            recent_dates.append(
                                Span(doc, date.start, date.end, label="relative_date")
                            )
                    elif value.direction.value == "past":
                        if (
                            -value.to_duration(
                                note_datetime=doc._.note_datetime,
                                infer_from_context=True,
                                tz="Europe/Paris",
                                default_day=15,
                            )
                            >= self.history_limit
                        ):
                            history_dates.append(
                                Span(doc, date.start, date.end, label="relative_date")
                            )
                        else:
                            recent_dates.append(
                                Span(doc, date.start, date.end, label="relative_date")
                            )
                elif value.mode == "absolute" and doc._.note_datetime:
                    try:
                        absolute_date = value.to_datetime(
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
                            value,
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

            close_recent_dates = []
            close_history_dates = []
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

            history_cues = [m for m in sub_matches if m.label_ == "history"]
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
                    token._.history = token._.history or history

            for ent in ents:
                ent._.history = ent._.history or history

                if self.explain:
                    ent._.history_cues += history_cues
                    ent._.recent_cues += recent_cues
                if not self.on_ents_only and ent._.history:
                    for token in ent:
                        token._.history = True

        return doc
