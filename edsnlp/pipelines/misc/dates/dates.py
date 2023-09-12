"""`eds.dates` pipeline."""
import warnings
from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span
from typing_extensions import Literal

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import (
    BaseNERComponent,
    SpanGetterArg,
    SpanSetterArg,
    get_spans,
    validate_span_getter,
)
from edsnlp.utils.filter import align_spans, filter_spans

from . import patterns
from .models import AbsoluteDate, Bound, Duration, Mode, Period, RelativeDate


class DatesMatcher(BaseNERComponent):
    """
    The `eds.dates` matcher detects and normalize dates within a medical document.
    We use simple regular expressions to extract date mentions.

    Scope
    -----
    The `eds.dates` pipeline finds absolute (eg `23/08/2021`) and relative (eg `hier`,
    `la semaine dernière`) dates alike. It also handles mentions of duration.

    | Type       | Example                       |
    | ---------- | ----------------------------- |
    | `absolute` | `3 mai`, `03/05/2020`         |
    | `relative` | `hier`, `la semaine dernière` |
    | `duration` | `pendant quatre jours`        |

    See the [tutorial](../../tutorials/detecting-dates.md) for a presentation of a
    full pipeline featuring the `eds.dates` component.

    ## Usage

    ```python
    import spacy

    import pendulum

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.dates")

    text = (
        "Le patient est admis le 23 août 2021 pour une douleur à l'estomac. "
        "Il lui était arrivé la même chose il y a un an pendant une semaine. "
        "Il a été diagnostiqué en mai 1995."
    )

    doc = nlp(text)

    dates = doc.spans["dates"]
    dates
    # Out: [23 août 2021, il y a un an, mai 1995]

    dates[0]._.date.to_datetime()
    # Out: 2021-08-23T00:00:00+02:00

    dates[1]._.date.to_datetime()
    # Out: None

    note_datetime = pendulum.datetime(2021, 8, 27, tz="Europe/Paris")

    dates[1]._.date.to_datetime(note_datetime=note_datetime)
    # Out: 2020-08-27T00:00:00+02:00

    date_2_output = dates[2]._.date.to_datetime(
        note_datetime=note_datetime,
        infer_from_context=True,
        tz="Europe/Paris",
        default_day=15,
    )
    date_2_output
    # Out: 1995-05-15T00:00:00+02:00

    doc.spans["durations"]
    # Out: [pendant une semaine]
    ```

    Extensions
    ----------
    The `eds.dates` pipeline declares two extensions on the `Span` object:

    - the `span._.date` attribute of a date contains a parsed version of the date.
    - the `span._.duration` attribute of a duration contains a parsed version of the
      duration.

    As with other components, you can use the `span._.value` attribute to get either the
    parsed date or the duration depending on the span.

    Parameters
    ----------
    nlp : Language
        The pipeline object
    name : Optional[str]
        Name of the pipeline component
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
    span_getter : SpanGetterArg
        Where to look for dates in the doc. By default, look in the whole doc. You can
        combine this with the `merge_mode` argument for interesting results.
    merge_mode : Literal["intersect", "align"]
        How to merge matched dates with the spans from `span_getter`, if given:

        - `intersect`: return only the matches that fall in the `span_getter` spans
        - `align`: if a date overlaps a span from `span_getter` (e.g. a date extracted
          by a machine learning model), return the `span_getter` span instead, and
          assign all the parsed information (`._.date` / `._.duration`) to it. Otherwise
          don't return the date.
    on_ents_only : Union[bool, str, Iterable[str]]
        Deprecated, use `span_getter` and `merge_mode` instead.
        Whether to look on dates in the whole document or in specific sentences:

        - If `True`: Only look in the sentences of each entity in doc.ents
        - If False: Look in the whole document
        - If given a string `key` or list of string: Only look in the sentences of
          each entity in `#!python doc.spans[key]`
    detect_periods : bool
        Whether to detect periods (experimental)
    detect_time: bool
        Whether to detect time inside dates
    period_proximity_threshold : int
        Max number of words between two dates to extract a period.
    as_ents : bool
        Deprecated, use span_setter instead.
        Whether to treat dates as entities
    attr : str
        spaCy attribute to use
    date_label : str
        Label to use for dates
    duration_label : str
        Label to use for durations
    period_label : str
        Label to use for periods
    span_setter : SpanSetterArg
        How to set matches in the doc.

    Authors and citation
    --------------------
    The `eds.dates` pipeline was developed by AP-HP's Data Science team.
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        name: str = "eds.dates",
        *,
        absolute: Optional[List[str]] = None,
        relative: Optional[List[str]] = None,
        duration: Optional[List[str]] = None,
        false_positive: Optional[List[str]] = None,
        on_ents_only: Union[bool, str, Iterable[str]] = False,
        span_getter: Optional[SpanGetterArg] = None,
        merge_mode: Literal["intersect", "align"] = "intersect",
        detect_periods: bool = False,
        detect_time: bool = True,
        period_proximity_threshold: int = 3,
        as_ents: bool = False,
        attr: str = "LOWER",
        date_label: str = "date",
        duration_label: str = "duration",
        period_label: str = "period",
        span_setter: SpanSetterArg = {
            "dates": ["date"],
            "durations": ["duration"],
            "periods": ["period"],
        },
    ):
        self.date_label = date_label
        self.duration_label = duration_label
        self.period_label = period_label

        # Backward compatibility
        if as_ents is True:
            warnings.warn(
                "The `as_ents` argument is deprecated."
                + (
                    " Pass `span_setter={} instead.".format(
                        {**span_setter, "ents": [self.date_label, self.duration_label]}
                    )
                    if isinstance(span_setter, dict)
                    else " Use the `span_setter` argument instead."
                ),
                DeprecationWarning,
            )
            span_setter = dict(span_setter)
            span_setter["ents"] = True
        super().__init__(nlp=nlp, name=name, span_setter=span_setter)

        if absolute is None:
            if detect_time:
                absolute = patterns.absolute_pattern_with_time
            else:
                absolute = patterns.absolute_pattern
        if relative is None:
            relative = patterns.relative_pattern
        if duration is None:
            duration = patterns.duration_pattern
        if false_positive is None:
            false_positive = patterns.false_positive_pattern

        if isinstance(absolute, str):
            absolute = [absolute]
        if isinstance(relative, str):
            relative = [relative]
        if isinstance(duration, str):
            relative = [duration]
        if isinstance(false_positive, str):
            false_positive = [false_positive]

        if on_ents_only:
            assert span_getter is None, (
                "Cannot use both `on_ents_only` and " "`span_getter`"
            )

            def span_getter(doc):
                return (span.sent for span in doc.ents)

            merge_mode = "intersect"
            warnings.warn(
                "The `on_ents_only` argument is deprecated."
                " Use the `span_getter` argument instead.",
                DeprecationWarning,
            )
        self.span_getter = validate_span_getter(span_getter, optional=True)
        self.merge_mode = merge_mode
        self.regex_matcher = RegexMatcher(attr=attr, alignment_mode="strict")

        self.regex_matcher.add("false_positive", false_positive)
        self.regex_matcher.add("absolute", absolute)
        self.regex_matcher.add("relative", relative)
        self.regex_matcher.add("duration", duration)

        self.detect_periods = detect_periods
        self.period_proximity_threshold = period_proximity_threshold

        self.as_ents = as_ents

        if detect_periods:
            logger.warning("The period extractor is experimental.")

        self.set_extensions()

    def set_extensions(self) -> None:
        """
        Set extensions for the dates pipeline.
        """

        if not Span.has_extension("datetime"):
            Span.set_extension("datetime", default=None)

        if not Span.has_extension(self.date_label):
            Span.set_extension(self.date_label, default=None)

        if not Span.has_extension(self.duration_label):
            Span.set_extension(self.duration_label, default=None)

        if not Span.has_extension(self.period_label):
            Span.set_extension(self.period_label, default=None)

    def process(self, doc: Doc) -> List[Tuple[Span, Dict[str, str]]]:
        """
        Find dates in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        dates:
            list of date spans
        """

        spans = None

        if self.span_getter is not None:
            spans = list(get_spans(doc, self.span_getter))
            matches = []
            for sent in set([ent.sent for ent in doc]):
                matches = chain(
                    matches,
                    self.regex_matcher(
                        sent,
                        as_spans=True,
                        return_groupdict=True,
                    ),
                )

        else:
            matches = self.regex_matcher(
                doc,
                as_spans=True,
                return_groupdict=True,
            )

        matches = filter_spans(matches)
        matches = [date for date in matches if date[0].label_ != "false_positive"]

        matches = list(self.parse(matches))

        if self.span_getter is not None:
            if self.merge_mode == "align":
                alignments = align_spans(matches, spans, sort_by_overlap=True)
                matches = []
                for span, aligned in zip(spans, alignments):
                    if len(aligned):
                        old = aligned[0]
                        span.label_ = old.label_
                        span._.set(self.date_label, old._.get(self.date_label))
                        span._.set(self.duration_label, old._.get(self.duration_label))
                        matches.append(span)

            elif self.merge_mode == "intersect":
                alignments = align_spans(matches, spans)
                matches = []
                for span, aligned in zip(spans, alignments):
                    matches.extend(aligned)
                matches = list(dict.fromkeys(matches))

        if self.detect_periods:
            matches.extend(self.process_periods(matches))

        return matches

    def parse(
        self, matches: List[Tuple[Span, Dict[str, str]]]
    ) -> Tuple[List[Span], List[Span]]:
        """
        Parse dates/durations using the groupdict returned by the matcher.

        Parameters
        ----------
        matches : List[Tuple[Span, Dict[str, str]]]
            List of tuples containing the spans and groupdict
            returned by the matcher.

        Returns
        -------
        Tuple[List[Span], List[Span]]
            List of processed spans, with the date parsed.
        """

        for span, groupdict in matches:
            if span.label_ == "relative":
                parsed = RelativeDate.parse_obj(groupdict)
                span.label_ = self.date_label
                span._.date = parsed
            elif span.label_ == "absolute":
                parsed = AbsoluteDate.parse_obj(groupdict)
                span.label_ = self.date_label
                span._.date = parsed
            else:
                parsed = Duration.parse_obj(groupdict)
                span.label_ = self.duration_label
                span._.duration = parsed

        return [span for span, _ in matches]

    def process_periods(self, dates: List[Span]) -> List[Span]:
        """
        Experimental period detection.

        Parameters
        ----------
        dates : List[Span]
            List of detected dates.

        Returns
        -------
        List[Span]
            List of detected periods.
        """

        if len(dates) < 2:
            return []

        periods = []
        seen = set()

        dates = list(sorted(dates, key=lambda d: d.start))

        for d1, d2 in zip(dates[:-1], dates[1:]):
            v1 = d1._.date if d1.label_ == self.date_label else d1._.duration
            v2 = d2._.date if d2.label_ == self.date_label else d2._.duration
            if v1.mode == Mode.DURATION or v2.mode == Mode.DURATION:
                pass
            elif d1 in seen or v1.bound is None or v2.bound is None:
                continue

            if (
                d1.end - d2.start < self.period_proximity_threshold
                and v1.bound != v2.bound
            ):
                period = Span(d1.doc, d1.start, d2.end, label=self.period_label)

                # If one date is a duration,
                # the other may not have a registered bound attribute.
                if v1.mode == Mode.DURATION:
                    m1 = Bound.FROM if v2.bound == Bound.UNTIL else Bound.UNTIL
                    m2 = v2.mode or Bound.FROM
                elif v2.mode == Mode.DURATION:
                    m1 = v1.mode or Bound.FROM
                    m2 = Bound.FROM if v1.bound == Bound.UNTIL else Bound.UNTIL
                else:
                    m1 = v1.mode or Bound.FROM
                    m2 = v2.mode or Bound.FROM

                period._.set(
                    self.period_label,
                    Period.parse_obj(
                        {
                            m1: d1,
                            m2: d2,
                        }
                    ),
                )

                seen.add(d1)
                seen.add(d2)

                periods.append(period)

        return periods

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags dates.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for dates
        """

        matches = self.process(doc)

        self.set_spans(doc, matches)

        return doc
