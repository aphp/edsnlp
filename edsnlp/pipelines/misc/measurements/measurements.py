import abc
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from itertools import repeat
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import regex
import spacy
from spacy.tokens import Doc, Span
from typing_extensions import TypedDict

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.misc.measurements.patterns import common_measurements
from edsnlp.utils.filter import filter_spans

__all__ = ["MeasurementsMatcher"]


AFTER_SNIPPET_LIMIT = 6
BEFORE_SNIPPET_LIMIT = 10


class UnitConfig(TypedDict):
    dim: str
    degree: int
    scale: float
    terms: List[str]
    followed_by: Optional[str] = None


class UnitlessRange(TypedDict):
    min: int
    max: int
    unit: str


class UnitlessPatternConfig(TypedDict):
    terms: List[str]
    ranges: List[UnitlessRange]


class UnitlessPatternConfigWithName(TypedDict):
    terms: List[str]
    ranges: List[UnitlessRange]
    name: str


class MeasureConfig(TypedDict):
    unit: str
    unitless_patterns: List[UnitlessPatternConfig]


class Measurement(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterable["SimpleMeasurement"]:
        """
        Iter over items of the measure (only one for SimpleMeasurement)

        Returns
        -------
        iterable : Iterable["SimpleMeasurement"]
        """

    @abc.abstractmethod
    def __getitem__(self, item) -> "SimpleMeasurement":
        """
        Access items of the measure (only one for SimpleMeasurement)

        Parameters
        ----------
        item : int

        Returns
        -------
        measure : SimpleMeasurement
        """


class UnitRegistry:
    def __init__(self, config: Dict[str, UnitConfig]):
        self.config = {unicodedata.normalize("NFKC", k): v for k, v in config.items()}
        for unit, unit_config in list(self.config.items()):
            if not unit.startswith("per_") and "per_" + unit not in unit_config:
                self.config["per_" + unit] = {
                    "dim": unit_config["dim"],
                    "degree": -unit_config["degree"],
                    "scale": 1 / unit_config["scale"],
                }

    @lru_cache(maxsize=-1)
    def parse_unit(self, unit: str) -> Tuple[str, float]:
        degrees = defaultdict(lambda: 0)
        scale = 1
        for part in regex.split("(?<!per)_", unit):
            unit_config = self.config[unicodedata.normalize("NFKC", part)]
            degrees[unit_config["dim"]] += unit_config["degree"]
            scale *= unit_config["scale"]
        return str(dict(sorted(degrees.items()))), scale


class SimpleMeasurement(Measurement):
    def __init__(self, value, unit, registry):
        """
        The SimpleMeasurement class contains the value and unit
        for a single non-composite measure

        Parameters
        ----------
        value : float
        unit : str
        """
        super().__init__()
        self.value = value
        self.unit = unit
        self.registry = registry

    def __iter__(self):
        return iter((self,))

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        return [self][item]

    def __str__(self):
        return f"{self.value} {self.unit}"

    def __repr__(self):
        return f"Measurement({self.value}, {repr(self.unit)})"

    def __eq__(self, other: Any):
        if isinstance(other, SimpleMeasurement):
            return self.convert_to(other.unit) == other.value
        return False

    def __add__(self, other: "SimpleMeasurement"):
        if other.unit == self.unit:
            return self.__class__(self.value + other.value, self.unit, self.registry)
        return self.__class__(
            self.value + other.convert_to(self.unit), self.unit, self.registry
        )

    def __lt__(self, other: "SimpleMeasurement"):
        return self.convert_to(other.unit) < other.value

    def __le__(self, other: "SimpleMeasurement"):
        return self.convert_to(other.unit) <= other.value

    def convert_to(self, other_unit):
        self_degrees, self_scale = self.registry.parse_unit(self.unit)
        other_degrees, other_scale = self.registry.parse_unit(other_unit)

        if self_degrees != other_degrees:
            raise AttributeError(
                f"Units {self.unit} and {other_unit} are not homogenous"
            )
        new_value = self.value * self_scale / other_scale
        return new_value

    def __getattr__(self, other_unit):
        return self.convert_to(other_unit)

    @classmethod
    def verify(cls, ent):
        return True


class MeasurementsMatcher:
    def __init__(
        self,
        nlp: spacy.Language,
        measurements: Union[List[str], Tuple[str], Dict[str, MeasureConfig]],
        units_config: Dict[str, UnitConfig],
        number_terms: Dict[str, List[str]],
        stopwords: List[str] = ("par", "sur", "de", "a", ":"),
        unit_divisors: List[str] = ("par", "/"),
        name: str = "measurements",
        ignore_excluded: bool = True,
        attr: str = "NORM",
    ):
        """
        Matcher component to extract measurements.
        A measurements is most often composed of a number and a unit like
        > 1,26 cm
        The unit can also be positioned in place of the decimal dot/comma
        > 1 cm 26
        Some measurements can be composite
        > 1,26 cm x 2,34 mm
        And sometimes they are factorized
        > Les trois kystes mesurent 1, 2 et 3cm.

        The recognized measurements are stored in the "measurements" SpanGroup.
        Each span has a `Measurement` object stored in the "value" extension attribute.

        Parameters
        ----------
        nlp : Language
            The SpaCy object.
        measurements : Dict[str, MeasureConfig]
            A mapping from measure names to MeasureConfig
            Each measure's configuration has the following shape:
            {
                "unit": str, # the unit of the measure (like "kg"),
                "unitless_patterns": { # optional patterns to handle unitless cases
                    "terms": List[str], # list of preceding terms used to trigger the
                    measure
                    # Mapping from ranges to unit to handle cases like
                    # ("Taille: 1.2" -> 1.20 m vs "Taille: 120" -> 120cm)
                    "ranges": List[{
                        "min": int,
                        "max": int,
                        "unit": str,
                    }, {
                        "min": int,
                        "unit": str,
                    }, ...],
                }
        number_terms: Dict[str, List[str]
            A mapping of numbers to their lexical variants
        stopwords: List[str]
            A list of stopwords that do not matter when placed between a unitless
            trigger
            and a number
        unit_divisors: List[str]
            A list of terms used to divide two units (like: m / s)
        attr : str
            Whether to match on the text ('TEXT') or on the normalized text ('NORM')
        ignore_excluded : bool
            Whether to exclude pollution patterns when matching in the text
        """

        if measurements is None:
            measurements = common_measurements
        elif isinstance(measurements, (list, tuple)):
            measurements = {m: common_measurements[m] for m in measurements}

        self.nlp = nlp
        self.name = name
        self.unit_registry = UnitRegistry(units_config)
        self.regex_matcher = RegexMatcher(attr=attr, ignore_excluded=True)
        self.term_matcher = EDSPhraseMatcher(nlp.vocab, attr=attr, ignore_excluded=True)
        self.unitless_patterns: Dict[str, UnitlessPatternConfigWithName] = {}
        self.unit_part_label_hashes: Set[int] = set()
        self.unitless_label_hashes: Set[int] = set()
        self.unit_followers: Dict[str, str] = {}
        self.measure_names: Dict[str, str] = {}

        # NUMBER PATTERNS
        self.regex_matcher.add(
            "number",
            [
                r"(?<![a-z-])\d+([ ]\d{3})*[ ]+(?:[,.][ ]+\d+)?",
                r"(?<![a-z-])\d+([ ]\d{3})*(?:[,.]\d+)?",
            ],
        )
        self.number_label_hashes = {nlp.vocab.strings["number"]}
        for number, terms in number_terms.items():
            self.term_matcher.build_patterns(nlp, {number: terms})
            self.number_label_hashes.add(nlp.vocab.strings[number])

        # UNIT PATTERNS
        for unit_name, unit_config in units_config.items():
            self.term_matcher.build_patterns(nlp, {unit_name: unit_config["terms"]})
            if unit_config.get("followed_by") is not None:
                self.unit_followers[unit_name] = unit_config["followed_by"]
            self.unit_part_label_hashes.add(nlp.vocab.strings[unit_name])

        self.unit_part_label_hashes.add(nlp.vocab.strings["per"])
        self.term_matcher.build_patterns(
            nlp,
            {
                "per": unit_divisors,
                "stopword": stopwords,
            },
        )

        # MEASURES
        for name, measure_config in measurements.items():
            unit = measure_config["unit"]
            self.measure_names[self.unit_registry.parse_unit(unit)[0]] = name
            if "unitless_patterns" in measure_config:
                for pattern in measure_config["unitless_patterns"]:
                    pattern_name = f"unitless_{len(self.unitless_patterns)}"
                    self.term_matcher.build_patterns(
                        nlp,
                        terms={
                            pattern_name: pattern["terms"],
                        },
                    )
                    self.unitless_label_hashes.add(nlp.vocab.strings[pattern_name])
                    self.unitless_patterns[pattern_name] = {"name": name, **pattern}

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:
        """
        Set extensions for the measurements pipeline.
        """

        if not Span.has_extension("value"):
            Span.set_extension("value", default=None)

    def extract_units(self, term_matches: Iterable[Span]) -> Iterable[Span]:
        """
        Extracts unit spans from the document by extracting unit atoms (declared in the
        units_config parameter) and aggregating them automatically
        Ex: "il faut 2 g par jour"
        => we extract [g]=unit(g), [par]=divisor(per), [jour]=unit(day)
        => we aggregate these adjacent matches together to compose a new unit g_per_day


        Parameters
        ----------
        term_matches: Iterable[Span]

        Returns
        -------
        Iterable[Span]
        """
        last = None
        units = []
        current = []
        unit_label_hashes = set()
        for unit_part in filter_spans(term_matches):
            if unit_part.label not in self.unit_part_label_hashes:
                continue
            if last is not None and unit_part.start != last.end and len(current):
                doc = current[0].doc
                # Last non "per" match: we don't want our units to be like `g_per`
                end = next(
                    (i for i, e in list(enumerate(current))[::-1] if e.label_ != "per"),
                    None,
                )
                if end is not None:
                    unit = "_".join(part.label_ for part in current[: end + 1])
                    units.append(Span(doc, current[0].start, current[end].end, unit))
                    unit_label_hashes.add(units[-1].label)
                current = []
                last = None
            if len(current) > 0 or unit_part.label_ != "per":
                current.append(unit_part)
            last = unit_part

        end = next(
            (i for i, e in list(enumerate(current))[::-1] if e.label_ != "per"), None
        )
        if end is not None:
            doc = current[0].doc
            unit = "_".join(part.label_ for part in current[: end + 1])
            units.append(Span(doc, current[0].start, current[end].end, unit))
            unit_label_hashes.add(units[-1].label)

        return units

    @classmethod
    def make_pseudo_sentence(
        cls,
        doc: Doc,
        matches: List[Tuple[Span, bool]],
        pseudo_mapping: Dict[int, str],
    ) -> Tuple[str, List[int]]:
        """
        Creates a pseudo sentence (one letter per entity)
        to extract higher order patterns
        Ex: the sentence
        "Il font {1}{,} {2} {et} {3} {cm} de long{.}" is transformed into "wn,n,nuw."

        Parameters
        ----------
        doc: Doc
        matches: List[(Span, bool)]
            List of tuple of span and whether the span represents a sentence end
        pseudo_mapping: Dict[int, str]
            A mapping from label to char in the pseudo sentence

        Returns
        -------
        (str, List[int])
            - the pseudo sentence
            - a list of offsets to convert match indices into pseudo sent char indices
        """
        pseudo = []
        last = 0
        offsets = []
        for ent, is_sent_split in matches:
            if ent.start != last:
                pseudo.append("w")
            offsets.append(len(pseudo))
            if is_sent_split:
                pseudo.append(".")
            else:
                pseudo.append(pseudo_mapping.get(ent.label, "w"))
            last = ent.end
        if len(doc) != last:
            pseudo.append("w")
        pseudo = "".join(pseudo)

        return pseudo, offsets

    def get_matches(self, doc):
        """
        Extract and filter regex and phrase matches in the document
        to prepare the measurement extraction.
        Returns the matches and a list of hashes to quickly find unit matches

        Parameters
        ----------
        doc: Doc

        Returns
        -------
        Tuple[List[(Span, bool)], Set[int]]
            - List of tuples of spans and whether the spans represents a sentence end
            - List of hash label to distinguish unit from other matches
        """
        sent_ends = [doc[i : i + 1] for i in range(len(doc)) if doc[i].is_sent_end]

        regex_matches = list(self.regex_matcher(doc, as_spans=True))
        term_matches = list(self.term_matcher(doc, as_spans=True))

        # Detect unit parts and compose them into units
        units = self.extract_units(term_matches)
        unit_label_hashes = {unit.label for unit in units}

        # Filter matches to prevent matches over dates or doc entities
        non_unit_terms = [
            term
            for term in term_matches
            if term.label not in self.unit_part_label_hashes
        ]

        # Filter out measurement-related spans that overlap already matched
        # entities (in doc.ents or doc.spans["dates"])
        # Note: we also include sentence ends tokens as 1-token spans in those matches
        spans__keep__is_sent_end = filter_spans(
            [
                # Tuples (span, keep = is measurement related, is sentence end)
                *zip(doc.spans.get("dates", ()), repeat(False), repeat(False)),
                *zip(regex_matches, repeat(True), repeat(False)),
                *zip(non_unit_terms, repeat(True), repeat(False)),
                *zip(units, repeat(True), repeat(False)),
                *zip(doc.ents, repeat(False), repeat(False)),
                *zip(sent_ends, repeat(True), repeat(True)),
            ],
            # filter entities to keep only the ...
            sort_key=measurements_match_tuples_sort_key,
        )

        # Remove non-measurement related spans (keep = False) and sort the matches
        matches_and_is_sentence_end: List[(Span, bool)] = sorted(
            [
                (span, is_sent_end)
                for span, keep, is_sent_end in spans__keep__is_sent_end
                # and remove entities that are not relevant to this pipeline
                if keep
            ]
        )

        return matches_and_is_sentence_end, unit_label_hashes

    def extract_measurements(self, doc: Doc):
        """
        Extracts measure entities from the document

        Parameters
        ----------
        doc: Doc

        Returns
        -------
        List[Span]
        """
        matches, unit_label_hashes = self.get_matches(doc)

        # Make match slice function to query them
        def get_matches_after(i):
            anchor = matches[i][0]
            for j, (ent, is_sent_end) in enumerate(matches[i + 1 :]):
                if not is_sent_end and ent.start > anchor.end + AFTER_SNIPPET_LIMIT:
                    return
                yield j + i + 1, ent

        def get_matches_before(i):
            anchor = matches[i][0]
            for j, (ent, is_sent_end) in enumerate(matches[i::-1]):
                if not is_sent_end and ent.end < anchor.start - BEFORE_SNIPPET_LIMIT:
                    return
                yield i - j, ent

        # Make a pseudo sentence to query higher order patterns in the main loop
        # `offsets` is a mapping from matches indices (ie match n°i) to
        # char indices in the pseudo sentence
        pseudo, offsets = self.make_pseudo_sentence(
            doc,
            matches,
            {
                self.nlp.vocab.strings["stopword"]: ",",
                self.nlp.vocab.strings["number"]: "n",
                **{name: "u" for name in unit_label_hashes},
                **{name: "n" for name in self.number_label_hashes},
            },
        )

        measurements = []
        matched_unit_indices = set()

        # Iterate through the number matches
        for number_idx, (number, is_sent_split) in enumerate(matches):
            if not is_sent_split and number.label not in self.number_label_hashes:
                continue

            # Detect the measure value
            try:
                if number.label_ == "number":
                    value = float(
                        number.text.replace(" ", "").replace(",", ".").replace(" ", "")
                    )
                else:
                    value = float(number.label_)
            except ValueError:
                continue

            unit_idx = unit_text = unit_norm = None

            # Find the closest unit after the number
            try:
                unit_idx, unit_text = next(
                    (j, ent)
                    for j, ent in get_matches_after(number_idx)
                    if ent.label in unit_label_hashes
                )
                unit_norm = unit_text.label_
            except (AttributeError, StopIteration):
                pass

            # Try to pair the number with this next unit if the two are only separated
            # by numbers and separators alternatively (as in [1][,] [2] [and] [3] cm)
            try:
                pseudo_sent = pseudo[offsets[number_idx] + 1 : offsets[unit_idx]]
                if not re.fullmatch(r"(,n)*", pseudo_sent):
                    unit_text, unit_norm = None, None
            except TypeError:
                pass

            # Otherwise, try to infer the unit from the preceding unit to handle cases
            # like (1 meter 50)
            if unit_norm is None and number_idx - 1 in matched_unit_indices:
                try:
                    unit_before = matches[number_idx - 1][0]
                    if unit_before.end == number.start:
                        unit_norm = self.unit_followers[unit_before.label_]
                except (KeyError, AttributeError, IndexError):
                    pass

            # If no unit was matched, try to detect unitless patterns before
            # the number to handle cases like ("Weight: 63, Height: 170")
            if not unit_norm:
                try:
                    (unitless_idx, unitless_text) = next(
                        (j, e)
                        for j, e in get_matches_before(number_idx)
                        if e.label in self.unitless_label_hashes
                    )
                    unit_norm = None
                    if re.fullmatch(
                        r"[,n]*",
                        pseudo[offsets[unitless_idx] + 1 : offsets[number_idx]],
                    ):
                        unitless_pattern = self.unitless_patterns[unitless_text.label_]
                        unit_norm = next(
                            scope["unit"]
                            for scope in unitless_pattern["ranges"]
                            if ("min" not in scope or value >= scope["min"])
                            and ("max" not in scope or value < scope["max"])
                        )
                except StopIteration:
                    pass

            # Otherwise, skip this number
            if not unit_norm:
                continue

            # Compute the final entity
            if unit_text and unit_text.end == number.start:
                ent = doc[unit_text.start : number.end]
            elif unit_text and unit_text.start == number.end:
                ent = doc[number.start : unit_text.end]
            else:
                ent = number

            # Compute the dimensionality of the parsed unit
            try:
                dims = self.unit_registry.parse_unit(unit_norm)[0]
            except KeyError:
                continue

            # If the measure was not requested, dismiss it
            # Otherwise, relabel the entity and create the value attribute
            if dims not in self.measure_names:
                continue

            ent._.value = SimpleMeasurement(value, unit_norm, self.unit_registry)
            ent.label_ = self.measure_names[dims]

            measurements.append(ent)

            if unit_idx is not None:
                matched_unit_indices.add(unit_idx)

        return measurements

    @classmethod
    def merge_adjacent_measurements(cls, measurements: List[Span]) -> List[Span]:
        """
        Aggregates extracted measurements together when they are adjacent to handle
        cases like
        - 1 meter 50 cm
        - 30° 4' 54"

        Parameters
        ----------
        measurements: List[Span]

        Returns
        -------
        List[Span]
        """
        merged = measurements[:1]
        for ent in measurements[1:]:
            last = merged[-1]

            if last.end == ent.start and last._.value.unit != ent._.value.unit:
                try:
                    new_value = last._.value + ent._.value
                    merged[-1] = last = last.doc[last.start : ent.end]
                    last._.value = new_value
                    last.label_ = ent.label_
                except (AttributeError, TypeError):
                    merged.append(ent)
            else:
                merged.append(ent)

        return merged

    def __call__(self, doc):
        """
        Adds measurements to document's "measurements" SpanGroup.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for extracted measurements.
        """
        measurements = self.extract_measurements(doc)
        measurements = self.merge_adjacent_measurements(measurements)

        doc.spans["measurements"] = measurements

        # for backward compatibility
        doc.spans["measures"] = doc.spans["measurements"]

        return doc


def measurements_match_tuples_sort_key(
    span__keep__is_sent_end: Tuple[Span, bool, bool]
) -> Tuple[int, int, bool]:
    span, _, is_sent_end = span__keep__is_sent_end

    length = span.end - span.start

    return length, span.end, not is_sent_end
