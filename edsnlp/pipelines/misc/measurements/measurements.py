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
from typing_extensions import Literal, NotRequired, TypedDict

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.base import (
    BaseNERComponent,
    SpanGetterArg,
    SpanSetterArg,
    get_spans,
    validate_span_getter,
)
from edsnlp.pipelines.misc.measurements import patterns
from edsnlp.utils.filter import align_spans, filter_spans, get_span_group

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
    min: NotRequired[int]
    max: NotRequired[int]
    unit: str


class UnitlessPatternConfig(TypedDict):
    terms: List[str]
    ranges: List[UnitlessRange]


class UnitlessPatternConfigWithName(TypedDict):
    terms: List[str]
    ranges: List[UnitlessRange]
    name: str


class MsrConfig(TypedDict):
    unit: str
    unitless_patterns: NotRequired[List[UnitlessPatternConfig]]
    name: NotRequired[str]


class Measurement(abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> Iterable["SimpleMeasurement"]:
        """
        Number of items in the measure (only one for SimpleMeasurement)

        Returns
        -------
        Iterable["SimpleMeasurement"]
        """

    @abc.abstractmethod
    def __iter__(self) -> Iterable["SimpleMeasurement"]:
        """
        Iter over items of the measure (only one for SimpleMeasurement)

        Returns
        -------
        Iterable["SimpleMeasurement"]
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
        SimpleMeasurement
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
        degrees = defaultdict(lambda: [])
        scale = 1
        for part in regex.split("(?<!per)_", unit):
            unit_config = self.config[unicodedata.normalize("NFKC", part)]
            degrees[unit_config["dim"]].append(unit_config["degree"])
            scale *= unit_config["scale"]
        degrees = {
            k: sum(v) if len(set(v)) > 1 else v[0]
            for k, v in degrees.items()
            if sum(v) != 0
        }
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

    def __len__(self):
        return 1

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

    def __lt__(self, other: Union["SimpleMeasurement", "RangeMeasurement"]):
        return self.convert_to(other.unit) < min((part.value for part in other))

    def __le__(self, other: Union["SimpleMeasurement", "RangeMeasurement"]):
        return self.convert_to(other.unit) <= min((part.value for part in other))

    def convert_to(self, other_unit):
        self_degrees, self_scale = self.registry.parse_unit(self.unit)
        other_degrees, other_scale = self.registry.parse_unit(other_unit)

        if self_degrees != other_degrees:
            raise AttributeError(
                f"Units {self.unit} and {other_unit} are not homogenous"
            )
        ratio = self_scale / other_scale
        if ratio != 1:
            return self.value * ratio
        return self.value

    def __getattr__(self, other_unit):
        return self.convert_to(other_unit)

    @classmethod
    def verify(cls, ent):
        return True


class RangeMeasurement(Measurement):
    def __init__(self, from_value, to_value, unit, registry):
        super().__init__()
        self.value = (from_value, to_value)
        self.unit = unit
        self.registry = registry

    @classmethod
    def from_measurements(cls, a, b):
        a_value = a.value
        b_value = b.convert_to(a.unit)
        return RangeMeasurement(a_value, b_value, a.unit, a.registry)

    def convert_to(self, other_unit):
        self_degrees, self_scale = self.registry.parse_unit(self.unit)
        other_degrees, other_scale = self.registry.parse_unit(other_unit)

        if self_degrees != other_degrees:
            raise AttributeError(
                f"Units {self.unit} and {other_unit} are not homogenous"
            )
        ratio = self_scale / other_scale
        if ratio == 1:
            return self.value
        from_value = self.value[0] * ratio
        to_value = self.value[1] * ratio
        return (from_value, to_value)

    def __iter__(self):
        yield self[0]
        yield self[1]

    def __len__(self):
        return 2

    def __lt__(self, other: Union[SimpleMeasurement, "RangeMeasurement"]):
        return max(self.convert_to(other.unit)) < min((part.value for part in other))

    def __le__(self, other: Union[SimpleMeasurement, "RangeMeasurement"]):
        return max(self.convert_to(other.unit)) <= max((part.value for part in other))

    def __getattr__(self, other_unit):
        return self.convert_to(other_unit)

    def __eq__(self, other: Any):
        if isinstance(other, RangeMeasurement):
            return self.convert_to(other.unit) == other.value
        return False

    def __str__(self):
        return f"{self.value[0]}-{self.value[1]} {self.unit}"

    def __repr__(self):
        return f"RangeMeasurement({self.value}, {repr(self.unit)})"

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        return SimpleMeasurement(self.value[item], self.unit, self.registry)

    @classmethod
    def verify(cls, ent):
        return True


class MeasurementsMatcher(BaseNERComponent):
    '''
    The `eds.measurements` matcher detects and normalizes numerical measurements
    within a medical document.

    !!! warning

        The ``measurements`` pipeline is still in active development and has not
        been rigorously validated. If you come across a measurement expression that
        goes undetected, please file an issue !

    Scope
    -----
    The `eds.measurements` matcher can extract simple (e.g. `3cm`) measurements.
    It can also detect elliptic enumerations (eg `32, 33 et 34kg`) of measurements
    of the same type and split the measurements accordingly.

    The normalized value can then be accessed via the `span._.{measure_name}` attribute,
    for instance `span._.size` or `span._.weight` and be converted on the fly to a
    desired unit. Like for other components, the `span._.value` extension can also be
    used to access the normalized value for any measurement span.


    The current matcher annotates the following measurements out of the box:

    | Measurement name | Example                |
    |------------------|------------------------|
    | `size`           | `1m50`, `1.50m`        |
    | `weight`         | `12kg`, `1kg300`       |
    | `bmi`            | `BMI: 24`, `24 kg.m-2` |
    | `volume`         | `2 cac`, `8ml`         |


    Examples
    --------
    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe(
        "eds.measurements",
        config=dict(
            measurements=["size", "weight", "bmi"],
            extract_ranges=True,
        ),
    )

    text = """
    Le patient est admis hier, fait 1m78 pour 76kg.
    Les deux nodules bénins sont larges de 1,2 et 2.4mm.
    BMI: 24.

    Le nodule fait entre 1 et 1.5 cm
    """

    doc = nlp(text)

    measurements = doc.spans["measurements"]

    measurements
    # Out: [1m78, 76kg, 1,2, 2.4mm, 24, entre 1 et 1.5 cm]

    measurements[0]
    # Out: 1m78

    str(measurements[0]._.size), str(measurements[0]._.value)
    # Out: ('1.78 m', '1.78 m')

    measurements[0]._.value.cm
    # Out: 178.0

    measurements[2]
    # Out: 1,2

    str(measurements[2]._.value)
    # Out: '1.2 mm'

    str(measurements[2]._.value.mm)
    # Out: 1.2

    measurements[4]
    # Out: 24

    str(measurements[4]._.value)
    # Out: '24 kg_per_m2'

    str(measurements[4]._.value.kg_per_m2)
    # Out: 24

    str(measurements[5]._.value)
    # Out: 1-1.5 cm
    ```

    To extract all sizes in centimeters, and average range measurements, you can
    use the following snippet:

    ```python
    sizes = [
        sum(item.cm for item in m._.value) / len(m._.value)
        for m in doc.spans["measurements"]
        if m.label_ == "size"
    ]
    sizes
    # Out: [178.0, 0.12, 0.24, 1.25]
    ```

    Customization
    -------------
    You can declare custom measurements by altering the patterns:

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe(
        "eds.measurements",
        config=dict(
            measurements={
                "my_custom_surface_measurement": {
                    # This measurement unit is homogenous to square meters
                    "unit": "m2",
                    # Handle cases like "surface: 1.8" (implied m2),
                    # vs "surface: 50" (implied cm2)
                    "unitless_patterns": [
                        {
                            "terms": ["surface", "aire"],
                            "ranges": [
                                {"unit": "m2", "min": 0, "max": 9},
                                {"unit": "cm2", "min": 10, "max": 100},
                            ],
                        }
                    ],
                },
            }
        ),
    )
    ```

    Extensions
    ----------
    The `eds.measurements` pipeline declares its extensions dynamically, depending
    on the `measurements` parameter: each measurement gets its own extension, and
    is assigned to a different span group.

    Parameters
    ----------
    nlp : Language
        The pipeline object
    name : str
        The name of the component.
    measurements : Union[str, List[Union[str, MsrConfig]], Dict[str, MsrConfig]]
        A mapping from measure names to MsrConfig
        Each measure's configuration has the following shape:
        ```{ .python .no-check }
        {
          # the unit (e.g. "kg"),
          "unit": str,
          "unitless_patterns": {
            # preceding trigger terms
            "terms": List[str],
            # unitless ranges -> unit patterns
            "ranges": List[
              {"min": int, "max": int, "unit": str},
              {"min": int, "unit": str},
              ...,
            ],
            ...
          }
        }
        ```
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
    compose_units: bool
        Whether to compose units (like "m/s" or "m.s-1")
    extract_ranges: bool
        Whether to extract ranges (like "entre 1 et 2 cm")
    range_patterns: List[Tuple[str, str]]
        A list of "{FROM} xx {TO} yy" patterns to match range measurements
    after_snippet_limit: int
        Maximum word distance after to link a part of a measurement after its number
    before_snippet_limit: int
        Maximum word distance after to link a part of a measurement before its number
    span_setter: Optional[SpanSetterArg]
        How to set the spans in the document. By default, each measurement will
        be assigned to its own span group (using either the "name" field of the
        config, or the key if you passed a dict), and to the "measurements" group.
    span_getter : SpanGetterArg
        Where to look for measurements in the doc. By default, look in the whole doc.
        You can combine this with the `merge_mode` argument for interesting results.
    merge_mode : Literal["intersect", "align"]
        How to merge matches with the spans from `span_getter`, if given:

        - `intersect`: return only the matches that fall in the `span_getter` spans
        - `align`: if a match overlaps a span from `span_getter` (e.g. a match
          extracted by a machine learning model), return the `span_getter` span
          instead, and assign all the parsed information (`._.date` / `._.duration`)
          to it. Otherwise, don't return the date.

    Authors and citation
    --------------------
    The `eds.measurements` pipeline was developed by AP-HP's Data Science team.
    '''

    # fmt: off
    def __init__(
          self,
          nlp: spacy.Language,
          name: str = "eds.measurements",
          *,
          measurements: Union[str, List[Union[str, MsrConfig]], Dict[str, MsrConfig]] = list(patterns.common_measurements.keys()),  # noqa: E501
          units_config: Dict[str, UnitConfig] = patterns.units_config,
          number_terms: Dict[str, List[str]] = patterns.number_terms,
          stopwords: List[str] = patterns.stopwords,
          unit_divisors: List[str] = patterns.unit_divisors,
          ignore_excluded: bool = True,
          compose_units: bool = True,
          attr: str = "NORM",
          extract_ranges: bool = False,
          range_patterns: List[Tuple[Optional[str], Optional[str]]] = patterns.range_patterns,  # noqa: E501
          after_snippet_limit: int = 6,
          before_snippet_limit: int = 10,
          span_getter: Optional[SpanGetterArg] = None,
          merge_mode: Literal["intersect", "align"] = "intersect",
          as_ents: bool = False,
          span_setter: Optional[SpanSetterArg] = None,
    ):
        # fmt: on
        if isinstance(measurements, str):
            measurements = [measurements]
        if isinstance(measurements, (list, tuple)):
            measurements = [
                m
                if isinstance(m, dict)
                else {**patterns.common_measurements[m], "name": m}
                for m in measurements
            ]
        elif isinstance(measurements, dict):
            measurements = [{"name": k, **m} for k, m in measurements.items()]

        self.nlp = nlp
        self.name = name
        self.unit_registry = UnitRegistry(units_config)
        self.unitless_patterns: Dict[str, UnitlessPatternConfigWithName] = {}
        self.unit_part_label_hashes: Set[int] = set()
        self.unitless_label_hashes: Set[int] = set()
        self.unit_followers: Dict[str, str] = {}
        self.measure_names: Dict[str, str] = {}
        self.compose_units = compose_units
        self.extract_ranges = extract_ranges
        self.range_patterns = range_patterns
        self.span_getter = (
            validate_span_getter(span_getter)
            if span_getter is not None
            else None
        )
        self.merge_mode = merge_mode
        self.before_snippet_limit = before_snippet_limit
        self.after_snippet_limit = after_snippet_limit

        # MEASURES
        for measure_config in measurements:
            name = measure_config["name"]
            unit = measure_config["unit"]
            self.measure_names[self.unit_registry.parse_unit(unit)[0]] = name

        if span_setter is None:
            span_setter = {
                "ents": as_ents,
                "measurements": True,
                **{
                    name: [name]
                    for name in self.measure_names.values()
                }
            }

        super().__init__(nlp=nlp, name=name, span_setter=span_setter)

        self.regex_matcher = RegexMatcher(
            attr=attr,
            ignore_excluded=True,
        )
        self.term_matcher = EDSPhraseMatcher(
            nlp.vocab,
            attr=attr,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=True,
        )
        for measure_config in measurements:
            name = measure_config["name"]
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

        # NUMBER PATTERNS
        one_plus = "[1-9][0-9]*"
        self.regex_matcher.add(
            "number",
            [
                rf"(?<![0-9][.,]?){one_plus}([ ]\d{{3}})*[ ]+(?:[,.][ ]+\d+)?",
                rf"(?<![0-9][.,]?){one_plus}([ ]\d{{3}})*(?:[,.]\d+)?",
                rf"(?<![0-9][.,]?){one_plus}([ ]/[ ]|/){one_plus}",
                r"(?<![0-9][.,]?)00?",
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
                "unitless_stopword": [":"],
            },
        )

    def set_extensions(self) -> None:
        """
        Set extensions for the measurements pipeline.
        """
        super().set_extensions()

        for name in self.measure_names.values():
            if not Span.has_extension(name):
                Span.set_extension(name, default=None)

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
            if last is not None and (
                  (
                        unit_part.doc[last.end: unit_part.start].text.strip() != ""
                        and len(current)
                  )
                  or (
                        not self.compose_units
                        and len(current)
                        and current[-1].label_ != "per"
                  )
            ):
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
          doclike: Union[Doc, Span],
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
        doclike: Union[Doc, Span]
            The document or span to transform
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
        snippet = doclike if isinstance(doclike, Span) else doclike[:]
        last = snippet.start
        offsets = []
        for ent, is_sent_split in matches:
            if (
                  ent.start != last
                  and not doclike.doc[last: ent.start].text.strip() == ""
            ):
                pseudo.append("w")
            offsets.append(len(pseudo))
            pseudo.append(pseudo_mapping.get(ent.label, "." if is_sent_split else "w"))
            last = ent.end
        if snippet.end != last and doclike.doc[last: snippet.end].text.strip() == "":
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
        sent_ends = [doc[i: i + 1] for i in range(len(doc)) if doc[i].is_sent_end]

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
        # Prevent from matching over ents that are not measurement related
        ents = (e for e in doc.ents if e.label_ not in self.measure_names.values())
        spans__keep__is_sent_end = filter_spans(
            [
                # Tuples (span, keep = is measurement related, is sentence end)
                *zip(get_span_group(doc, "dates"), repeat(False), repeat(False)),
                *zip(regex_matches, repeat(True), repeat(False)),
                *zip(non_unit_terms, repeat(True), repeat(False)),
                *zip(units, repeat(True), repeat(False)),
                *zip(ents, repeat(False), repeat(False)),
                *zip(sent_ends, repeat(True), repeat(True)),
            ]
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

    def extract_measurements(self, doclike: Doc):
        """
        Extracts measure entities from the document

        Parameters
        ----------
        doclike: Doc

        Returns
        -------
        List[Span]
        """
        doc = doclike.doc if isinstance(doclike, Span) else doclike
        matches, unit_label_hashes = self.get_matches(doclike)

        # Make match slice function to query them
        def get_matches_after(i):
            anchor = matches[i][0]
            for j, (ent, is_sent_end) in enumerate(matches[i + 1:]):
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
            doclike,
            matches,
            {
                self.nlp.vocab.strings["stopword"]: ",",
                self.nlp.vocab.strings["unitless_stopword"]: ":",
                self.nlp.vocab.strings["number"]: "n",
                **{name: "u" for name in unit_label_hashes},
                **{name: "n" for name in self.number_label_hashes},
            },
        )

        measurements = []
        matched_unit_indices = set()
        matched_number_indices = set()

        # Iterate through the number matches
        for number_idx, (number, is_sent_split) in enumerate(matches):
            if not is_sent_split and number.label not in self.number_label_hashes:
                continue

            # Detect the measure value
            try:
                if number.label_ == "number":
                    number_text = (
                        number.text.replace(" ", "")
                        .replace(",", ".")
                        .replace(" ", "")
                        .lstrip("0")
                    )
                    value = eval(number_text or "0")
                else:
                    value = eval(number.label_)
            except (ValueError, SyntaxError):
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
                pseudo_sent = pseudo[offsets[number_idx] + 1: offsets[unit_idx]]
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
                          r"[,:n]*",
                          pseudo[offsets[unitless_idx] + 1: offsets[number_idx]],
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
            # TODO: handle this part better without .text.strip(), with cases for
            #  stopwords, etc
            if (
                  unit_text
                  and number.start <= unit_text.end
                  and doc[number.end: unit_text.start].text.strip() == ""
            ):
                ent = doc[number.start: unit_text.end]
            elif (
                  unit_text
                  and unit_text.start <= number.end
                  and doc[unit_text.end: number.start].text.strip() == ""
            ):
                ent = doc[unit_text.start: number.end]
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

            ent.label_ = self.measure_names[dims]
            ent._.set(
                self.measure_names[dims],
                SimpleMeasurement(value, unit_norm, self.unit_registry)
            )

            measurements.append(ent)

            if unit_idx is not None:
                matched_unit_indices.add(unit_idx)
            if number_idx is not None:
                matched_number_indices.add(number_idx)

        unmatched = []
        for idx, (match, _) in enumerate(matches):
            if (
                  match.label in unit_label_hashes
                  and idx not in matched_unit_indices
                  or match.label in self.number_label_hashes
                  and idx not in matched_number_indices
            ):
                unmatched.append(match)

        return measurements, unmatched

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
                    merged[-1] = last = last.doc[last.start: ent.end]
                    last.label_ = ent.label_
                    last._.set(last.label_, new_value)
                except (AttributeError, TypeError):
                    merged.append(ent)
            else:
                merged.append(ent)

        return merged

    def merge_measurements_in_ranges(self, measurements: List[Span]) -> List[Span]:
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
        if not self.extract_ranges or not self.range_patterns:
            return measurements

        merged = measurements[:1]
        for ent in measurements[1:]:
            last = merged[-1]

            from_text = last.doc[last.start - 1].norm_ if last.start > 0 else None
            to_text = get_text(last.doc[last.end: ent.start], "NORM", True)
            matching_patterns = [
                (a, b)
                for a, b in self.range_patterns
                if b == to_text and (a is None or a == from_text)
            ]
            if len(matching_patterns):
                try:
                    new_value = RangeMeasurement.from_measurements(
                        last._.value, ent._.value
                    )
                    merged[-1] = last = last.doc[
                                        last.start
                                        if matching_patterns[0][0] is None
                                        else last.start - 1: ent.end
                                        ]
                    last.label_ = ent.label_
                    last._.set(last.label_, new_value)
                except (AttributeError, TypeError):
                    merged.append(ent)
            else:
                merged.append(ent)

        return merged

    def merge_with_existing(
          self,
          extracted: List[Span],
          existing: List[Span],
    ) -> List[Span]:
        """
        Merges the extracted measurements with the existing measurements in the
        document.

        Parameters
        ----------
        extracted: List[Span]
            The extracted measurements
        existing: List[Span]
            The existing measurements in the document

        Returns
        -------
        List[Span]
        """
        if self.merge_mode == "align":
            spans_measurements = align_spans(extracted, existing, sort_by_overlap=True)

            extracted = []
            for span, span_measurements in zip(existing, spans_measurements):
                if len(span_measurements):
                    span._.set(span.label_, span_measurements[0]._.get(span.label_))
                    extracted.append(span)

        elif self.merge_mode == "intersect":
            spans_measurements = align_spans(extracted, existing)
            extracted = []
            for span, span_measurements in zip(existing, spans_measurements):
                extracted.extend(span_measurements)
            extracted = list(dict.fromkeys(extracted))

        return extracted

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
        existing = (
            list(get_spans(doc, self.span_getter))
            if self.span_getter is not None
            else ())
        snippets = (
            dict.fromkeys(ent.sent for ent in existing)
            if self.span_getter is not None
            else [doc]
        )
        measurements = [m for s in snippets for m in self.extract_measurements(s)[0]]
        measurements = self.merge_adjacent_measurements(measurements)
        measurements = self.merge_measurements_in_ranges(measurements)
        if self.span_getter is not None:
            measurements = self.merge_with_existing(measurements, existing)

        self.set_spans(doc, measurements)

        return doc
