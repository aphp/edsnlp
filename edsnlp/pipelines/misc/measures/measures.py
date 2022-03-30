import abc
from typing import Callable, Dict, Iterable, List, Tuple, Union

import regex
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans


def disj_capture(regexes, capture=True):
    return "|".join(
        ("(?P<{key}>{forms})" if capture else "{forms}").format(
            key=key, forms="|".join(forms)
        )
        for key, forms in regexes.items()
    )


def rightmost_largest_sort_key(span):
    return span.end, (len(span))


def make_patterns(measure: "Measure") -> Dict[str, Union[List[str], str]]:
    """
    Build recognition and extraction patterns for a given Measure class

    Parameters
    ----------
    measure: Measure class
        The measure to build recognition and extraction patterns for

    Returns
    -------
    trigger : List[str]
    extraction : str
    """
    unit_prefix_reg = disj_capture(
        {key: [entry["prefix"]] for key, entry in measure.UNITS.items()},
        capture=True,
    )
    unit_abbreviation_reg = disj_capture(
        {key: [entry["abbr"]] for key, entry in measure.UNITS.items()},
        capture=True,
    )
    unit_reg = rf"(?:(?:{unit_prefix_reg})[a-z]*|(?:{unit_abbreviation_reg})(?![a-z]))"

    number_reg = rf"(?:{measure.INTEGER}(?:[,.]{measure.INTEGER})?)"
    infix_measure_reg = rf"(?:{measure.INTEGER}{unit_reg}{measure.INTEGER})"

    # Simple measure
    simple_measure_reg = rf"{number_reg}\s*{unit_reg}"
    trigger = [
        simple_measure_reg,
        infix_measure_reg,
        # Factorized measures separated by a conjunction
        rf"{number_reg}(?=(?:\s*[,]\s*{number_reg})*\s*"
        rf"(?:{measure.CONJUNCTIONS})\s*{number_reg}\s*{unit_reg})",
    ]
    if measure.COMPOSITE:
        # Factorized composite measures (3 x 2cm)
        trigger.append(
            rf"(?<![a-z]){number_reg}"
            rf"(?:\s*(?:{measure.COMPOSERS})\s*{number_reg})*\s*{unit_reg}"
        )
        # Expanded composite measures (3cm x 2cm)
        trigger.append(
            rf"(?<![a-z])(?:{infix_measure_reg}|{simple_measure_reg})"
            rf"(\s*(?:{measure.COMPOSERS})\s*"
            rf"(?:{infix_measure_reg}|{simple_measure_reg}))*"
        )

    unit_reg_capture = (
        rf"(?:(?:{unit_prefix_reg})[a-z]*|(?:{unit_abbreviation_reg})(?![a-z]))"
    )

    return {
        "trigger": trigger,
        "extraction": rf"(?P<int_part>{measure.INTEGER})\s*(?:[,.]|"
        rf"{unit_reg_capture})?\s*(?P<dec_part>{measure.INTEGER})?",
    }


def make_simple_getter(name):
    def getter(self):
        """
        Get a scaled numerical value of a measure

        Parameters
        ----------
        self

        Returns
        -------
        float
        """
        return self.value * self._get_scale_to(name)

    return getter


def make_multi_getter(name: str) -> Callable[["CompositeMeasure"], Tuple[float]]:
    def getter(self) -> Tuple[float]:
        """
        Get a scaled numerical values of a multi-measure

        Parameters
        ----------
        self

        Returns
        -------
        float
        """
        return tuple(getattr(measure, name) for measure in self.measures)

    return getter


class Measure(abc.ABC):
    INTEGER = r"(?:[0-9]+)"
    CONJUNCTIONS = "et|ou"
    COMPOSERS = r"[x*]|par"

    UNITS = {}
    COMPOSITE = None

    @abc.abstractmethod
    def __iter__(self) -> Iterable["SimpleMeasure"]:
        """
        Iter over items of the measure (only one for SimpleMeasure)

        Returns
        -------
        iterable : Iterable["SimpleMeasure"]
        """

    @abc.abstractmethod
    def __getitem__(self, item) -> "SimpleMeasure":
        """
        Access items of the measure (only one for SimpleMeasure)

        Parameters
        ----------
        item : int

        Returns
        -------
        measure : SimpleMeasure
        """


class SimpleMeasure(Measure):
    def __init__(self, value, unit):
        """
        The SimpleMeasure class contains the value and unit
        for a single non-composite measure

        Parameters
        ----------
        value : float
        unit : str
        """
        super().__init__()
        self.value = value
        self.unit = unit

    @classmethod
    @abc.abstractmethod
    def parse(
        self, int_part: str, dec_part: str, unit: str, infix: bool
    ) -> "SimpleMeasure":
        """
        Class method to create an instance from the match groups

        int_part : str
            The integer part of the match (eg 12 in 12 metres 50 or 12.50metres)
        dec_part : str
            The decimal part of the match (eg 50 in 12 metres 50 or 12.50metres)
        unit : str
            The normalized variant of the unit (eg "m" for 12 metre 50)
        infix : bool
            Whether the unit was in the before (True) or after (False) the decimal part
        """

    def _get_scale_to(self, unit: str):
        return self.UNITS[self.unit]["value"] / self.UNITS[unit]["value"]

    def __iter__(self):
        return iter((self,))

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        return [self][item]

    def __str__(self):
        return f"{self.value}{self.unit}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value}, {repr(self.unit)})"

    def __eq__(self, other: "SimpleMeasure"):
        return getattr(self, other.unit) == other.value

    def __lt__(self, other: "SimpleMeasure"):
        return getattr(self, other.unit) < other.value

    def __le__(self, other: "SimpleMeasure"):
        return getattr(self, other.unit) <= other.value


class CompositeMeasure(Measure):
    """
    The CompositeMeasure class contains a sequence
    of multiple SimpleMeasure instances

    Parameters
    ----------
    measures : List[SimpleMeasure]
    """

    def __init__(self, measures: Iterable["SimpleMeasure"]):
        super().__init__()
        self.measures = list(measures)

    def __iter__(self):
        return iter(self.measures)

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        res = self.measures[item]
        return res

    def __str__(self):
        return " x ".join(map(str, self.measures))

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.measures)})"


class Measures(BaseComponent):
    """
    Matcher component to extract measures.
    A measures is most often composed of a number and a unit like
    > 1,26 cm
    The unit can also be positioned in place of the decimal dot/comma
    > 1 cm 26
    Some measures can be composite
    > 1,26 cm x 2,34 mm
    And sometimes they are factorized
    > Les trois kystes mesurent 1, 2 et 3cm.

    The recognized measures are stored in the "measures" SpanGroup.
    Each span has a `Measure` object stored in the "value" extension attribute.

    Parameters
    ----------
    nlp : Language
        The SpaCy object.
    measures : List[str]
        The registry names of the measures to extract
    attr : str
        Whether to match on the text ('TEXT') or on the normalized text ('NORM')
    ignore_excluded : bool
        Whether to exclude pollution patterns when matching in the text
    """

    def __init__(
        self,
        nlp: Language,
        measures: List[str],
        attr: str,
        ignore_excluded: bool,
    ):

        self.regex_matcher = RegexMatcher(
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        self.extraction_regexes = {}
        self.measures: Dict[str, Measure] = {}
        for name in measures:
            cls: Measure = spacy.registry.misc.get(name)
            self.measures[name] = cls
            regexes = make_patterns(cls)
            self.regex_matcher.add(name, regexes["trigger"])
            self.extraction_regexes[name] = regexes["extraction"]

        self.set_extensions()

    @staticmethod
    def set_extensions() -> None:
        super(Measures, Measures).set_extensions()
        if not Span.has_extension("value"):
            Span.set_extension("value", default=None)

    def __call__(self, doc: Doc) -> Doc:
        """
        Adds measures to document's "measures" SpanGroup.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for extracted terms.
        """

        matches = dict(self.regex_matcher(doc, as_spans=True, return_groupdict=True))

        # Filter spans by rightmost, largest spans first to handle cases like 1 m 50 kg
        # while keeping the corresponding groupdicts
        matches = {
            match: matches[match]
            for match in filter_spans(matches, sort_key=rightmost_largest_sort_key)
        }

        measures = []
        for match, groupdict in matches.items():
            measure_name = match.label_
            extraction_regex = self.extraction_regexes[measure_name]

            parsed_values = []

            shared_unit_part = next(
                (key for key, val in groupdict.items() if val is not None), None
            )
            for sub_match in regex.finditer(extraction_regex, match.text):
                sub_groupdict = dict(sub_match.groupdict())

                # Integer part of the match
                int_part = sub_groupdict.pop("int_part", 0)

                # Decimal part of the match, if any
                dec_part = sub_groupdict.pop("dec_part", 0) or 0

                # If the unit was not postfix (in cases like 1cm, or 1 et 2cm)
                # the unit must be infix: we extract it now using non empty groupdict
                # entries
                infix_unit_part = next(
                    (key for key, val in sub_groupdict.items() if val is not None),
                    None,
                )
                unit_part = infix_unit_part or shared_unit_part

                # Create one SimpleMeasure per submatch inside each match...
                parsed_values.append(
                    self.measures[measure_name].parse(
                        int_part=int_part,
                        dec_part=dec_part,
                        unit=unit_part,
                        infix=infix_unit_part is not None,
                    )
                )

            # ... and compose theses measures together if there are more than one
            measure = Span(doc, start=match.start, end=match.end, label=measure_name)
            measure._.value = (
                parsed_values[0]
                if len(parsed_values) == 1
                else self.measures[measure_name].COMPOSITE(parsed_values)
                if self.measures[measure_name].COMPOSITE is not None
                else parsed_values[-1]
            )
            measures.append(match)

        doc.spans["measures"] = sorted(measures)

        return doc
