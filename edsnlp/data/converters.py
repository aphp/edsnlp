"""
Converters are used to convert documents between python dictionaries and Doc objects.
There are two types of converters: readers and writers. Readers convert dictionaries to
Doc objects, and writers convert Doc objects to dictionaries.
"""

import inspect
import warnings
from copy import copy
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pydantic
import spacy
from confit.registry import ValidatedFunction
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Span

import edsnlp
from edsnlp import registry
from edsnlp.core.stream import CONTEXT
from edsnlp.utils.bindings import BINDING_GETTERS
from edsnlp.utils.span_getters import (
    SpanGetterArg,
    SpanSetterArg,
    get_spans,
    get_spans_with_group,
    set_spans,
)
from edsnlp.utils.typing import AsList, Validated

FILENAME = "__FILENAME__"
SPAN_BUILTIN_ATTRS = ("sent", "label_", "kb_id_", "text")

SCHEMA = {}

_DEFAULT_TOKENIZER = None

# For backward compatibility
SequenceStr = AsList[str]


def without_filename(d):
    d.pop(FILENAME, None)
    return d


def validate_kwargs(func, kwargs):
    if (
        hasattr(func, "__call__")
        and not hasattr(func, "__defaults__")
        and hasattr(func.__call__, "__self__")
    ):
        func = func.__call__
    has_self = restore = False
    spec = inspect.getfullargspec(func)
    try:
        if hasattr(func, "__func__"):
            has_self = hasattr(func, "__self__")
            func = func.__func__.__get__(None, func.__func__.__class__)
            old_annotations = func.__annotations__
            old_defaults = func.__defaults__
            restore = True
            func.__annotations__ = copy(func.__annotations__)
            func.__annotations__[spec.args[0]] = Optional[Any]
            func.__annotations__[spec.args[1]] = Optional[Any]
            func.__defaults__ = (
                None,
                None,
                *(spec.defaults or ())[-len(spec.args) + 2 :],
            )
        else:
            func: FunctionType = copy(func)
            old_annotations = func.__annotations__
            old_defaults = func.__defaults__
            restore = True
            func.__annotations__[spec.args[0]] = Optional[Any]
            func.__defaults__ = (None, *(spec.defaults or ())[-len(spec.args) + 1 :])
        vd = ValidatedFunction(func, {"arbitrary_types_allowed": True})
        model = vd.init_model_instance(
            **{k: v for k, v in kwargs.items() if k in spec.args}
        )
        fields = model.__fields__ if pydantic.__version__ < "2" else model.model_fields
        d = {
            k: v
            for k, v in model.__dict__.items()
            if (k in fields or fields[k].default_factory)
        }
        d.pop("v__duplicate_kwargs", None)  # see pydantic ValidatedFunction code
        d.pop(vd.v_args_name, None)
        d.pop(spec.args[0], None)
        if has_self:
            d.pop(spec.args[1], None)
        return {**(d.pop(vd.v_kwargs_name, None) or {}), **d}
    finally:
        if restore:
            func.__annotations__ = old_annotations
            func.__defaults__ = old_defaults


class AttributesMappingArg(Validated):
    """
    A span attribute mapping (can be a list too to keep the same names).

    For instance:

    - `doc_attributes="note_datetime"` will map the `note_datetime` JSON attribute to
      the `note_datetime` extension.
    - `span_attributes=["negation", "family"]` will map the `negation` and `family` JSON
      attributes to the `negation` and `family` extensions.
    """

    @classmethod
    def validate(cls, value, config=None) -> Dict[str, str]:
        return validate_attributes_mapping(value)


if TYPE_CHECKING:
    AttributesMappingArg = Union[str, Sequence[str], Dict[str, str]]  # noqa: F811


def validate_attributes_mapping(value: AttributesMappingArg) -> Dict[str, str]:
    if isinstance(value, str):
        return {value: value}
    if isinstance(value, list):
        return {item: item for item in value}
    else:
        return value


def get_current_tokenizer():
    global _DEFAULT_TOKENIZER
    if "tokenizer" in CONTEXT[0]:
        return CONTEXT[0]["tokenizer"]
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = edsnlp.blank("eds").tokenizer
    return _DEFAULT_TOKENIZER


@registry.factory.register("eds.standoff_dict2doc", spacy_compatible=False)
class StandoffDict2DocConverter:
    """
    !!! note "Why does BRAT/Standoff need a converter ?"

        You may wonder : why do I need a converter ? Since BRAT is already a NLP
        oriented format, it should be straightforward to convert it to a Doc object.

        Indeed, we do provide a default converter for the BRAT standoff format, but we
        also acknowledge that there may be more than one way to convert a standoff
        document to a Doc object. For instance, an annotated span may be used to
        represent a relation between two smaller included entities, or another entity
        scope, etc.

        In such cases, we recommend you use a custom converter as described
        [here](/data/converters/#custom-converter).

    Examples
    --------

    ```{ .python .no-check }
    # Any kind of reader (`edsnlp.data.read/from_...`) can be used here
    docs = edsnlp.data.read_standoff(
        "path/to/standoff",
        converter="standoff",  # set by default

        # Optional parameters
        tokenizer=tokenizer,
        span_setter={"ents": True, "*": True},
        span_attributes={"negation": "negated"},
        keep_raw_attribute_values=False,
        default_attributes={"negated": False, "temporality": "present"},
    )
    ```

    Parameters
    ----------
    nlp: Optional[PipelineProtocol]
        The pipeline object (optional and likely not needed, prefer to use the
        `tokenizer` directly argument instead).
    tokenizer: Optional[Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_pipeline` in a
          [Stream][edsnlp.core.stream.Stream].
        - or the `eds` tokenizer by default.
    span_setter : SpanSetterArg
        The span setter to use when setting the spans in the documents. Defaults to
        setting the spans in the `ents` attribute, and creates a new span group for
        each JSON entity label.
    span_attributes : Optional[AttributesMappingArg]
        Mapping from BRAT attributes to Span extensions (can be a list too).
        By default, all attributes are imported as Span extensions with the same name.
    keep_raw_attribute_values : bool
        Whether to keep the raw attribute values (as strings) or to convert them to
        Python objects (e.g. booleans).
    default_attributes : AttributesMappingArg
        How to set attributes on spans for which no attribute value was found in the
        input format. This is especially useful for negation, or frequent attributes
        values (e.g. "negated" is often False, "temporal" is often "present"), that
        annotators may not want to annotate every time.
    notes_as_span_attribute : Optional[str]
        If set, the AnnotatorNote annotations will be concatenated and stored in a span
        attribute with this name.
    split_fragments : bool
        Whether to split the fragments into separate spans or not. If set to False, the
        fragments will be concatenated into a single span.
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[Tokenizer] = None,
        span_setter: SpanSetterArg = {"ents": True, "*": True},
        span_attributes: Optional[AttributesMappingArg] = None,
        keep_raw_attribute_values: bool = False,
        bool_attributes: AsList[str] = [],
        default_attributes: AttributesMappingArg = {},
        notes_as_span_attribute: Optional[str] = None,
        split_fragments: bool = True,
    ):
        self.tokenizer = tokenizer
        self.span_setter = span_setter
        self.span_attributes = span_attributes  # type: ignore
        self.keep_raw_attribute_values = keep_raw_attribute_values
        self.default_attributes = default_attributes
        self.notes_as_span_attribute = notes_as_span_attribute
        self.split_fragments = split_fragments
        for attr in bool_attributes:
            self.default_attributes[attr] = False

    def __call__(self, obj, tokenizer=None):
        # tok = get_current_tokenizer() if self.tokenizer is None else self.tokenizer
        tok = tokenizer or self.tokenizer or get_current_tokenizer()
        doc = tok(obj["text"] or "")
        doc._.note_id = obj.get("doc_id", obj.get(FILENAME))

        spans = []

        for dst in (
            *(() if self.span_attributes is None else self.span_attributes.values()),
            *self.default_attributes,
        ):
            if not Span.has_extension(dst):
                Span.set_extension(dst, default=None)

        for ent in obj.get("entities") or ():
            fragments = (
                [
                    {
                        "begin": min(f["begin"] for f in ent["fragments"]),
                        "end": max(f["end"] for f in ent["fragments"]),
                    }
                ]
                if not self.split_fragments
                else ent["fragments"]
            )
            for fragment in fragments:
                span = doc.char_span(
                    fragment["begin"],
                    fragment["end"],
                    label=ent["label"],
                    alignment_mode="expand",
                )
                attributes = (
                    {a["label"]: a["value"] for a in ent["attributes"]}
                    if isinstance(ent["attributes"], list)
                    else ent["attributes"]
                )
                if self.notes_as_span_attribute and ent["notes"]:
                    ent["attributes"][self.notes_as_span_attribute] = "|".join(
                        note["value"] for note in ent["notes"]
                    )
                for label, value in attributes.items():
                    new_name = (
                        self.span_attributes.get(label, None)
                        if self.span_attributes is not None
                        else label
                    )
                    if self.span_attributes is None and not Span.has_extension(
                        new_name
                    ):
                        Span.set_extension(new_name, default=None)

                    if new_name:
                        value = True if value is None else value
                        if not self.keep_raw_attribute_values:
                            value = (
                                True
                                if value in ("True", "true")
                                else False
                                if value in ("False", "false")
                                else value
                            )
                        span._.set(new_name, value)

                spans.append(span)

        set_spans(doc, spans, span_setter=self.span_setter)
        for attr, value in self.default_attributes.items():
            for span in spans:
                if span._.get(attr) is None:
                    span._.set(attr, value)

        return doc


@registry.factory.register("eds.standoff_doc2dict", spacy_compatible=False)
class StandoffDoc2DictConverter:
    """
    Examples
    --------

    ```{ .python .no-check }
    # Any kind of writer (`edsnlp.data.read/from_...`) can be used here
    edsnlp.data.write_standoff(
        docs,
        converter="standoff",  # set by default

        # Optional parameters
        span_getter={"ents": True},
        span_attributes=["negation"],
    )
    # or docs.to_standoff(...) if it's already a
    # [Stream][edsnlp.core.stream.Stream]
    ```

    Parameters
    ----------
    span_getter: SpanGetterArg
        The span getter to use when getting the spans from the documents. Defaults to
        getting the spans in the `ents` attribute.
    span_attributes: AttributesMappingArg
        Mapping from Span extensions to JSON attributes (can be a list too).
        By default, no attribute is exported, except `note_id`.
    """

    def __init__(
        self,
        *,
        span_getter: Optional[SpanGetterArg] = {"ents": True},
        span_attributes: AttributesMappingArg = {},
    ):
        self.span_getter = span_getter
        self.span_attributes = span_attributes

    def __call__(self, doc):
        spans = get_spans(doc, self.span_getter)
        span_binding_getters = {
            obj_name: BINDING_GETTERS[
                ("_." + ext_name)
                if ext_name.split(".")[0] not in SPAN_BUILTIN_ATTRS
                else ext_name
            ]
            for ext_name, obj_name in self.span_attributes.items()
        }
        obj = {
            FILENAME: doc._.note_id,
            "doc_id": doc._.note_id,
            "text": doc.text,
            "entities": [
                {
                    "entity_id": i,
                    "fragments": [
                        {
                            "begin": ent.start_char,
                            "end": ent.end_char,
                        }
                    ],
                    "attributes": {
                        obj_name: value
                        for obj_name, value in (
                            (k, getter(ent))
                            for k, getter in span_binding_getters.items()
                        )
                        if value is not None
                    },
                    "label": ent.label_,
                }
                for i, ent in enumerate(sorted(dict.fromkeys(spans)))
            ],
        }
        return obj


@registry.factory.register("eds.conll_dict2doc", spacy_compatible=False)
class ConllDict2DocConverter:
    """
    TODO
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.tokenizer = tokenizer

    def __call__(self, obj, tokenizer=None):
        tok = get_current_tokenizer() if self.tokenizer is None else self.tokenizer
        vocab = tok.vocab
        words_data = [word for word in obj["words"] if "-" not in word["ID"]]
        words = [word["FORM"] for word in words_data]
        spaces = ["SpaceAfter=No" not in w.get("MISC", "") for w in words_data]
        doc = Doc(vocab, words=words, spaces=spaces)

        id_to_word = {word["ID"]: i for i, word in enumerate(words_data)}
        for word_data, word in zip(words_data, doc):
            for key, value in word_data.items():
                if key in ("ID", "FORM", "MISC"):
                    pass
                elif key == "LEMMA":
                    word.lemma_ = value
                elif key == "UPOS":
                    word.pos_ = value
                elif key == "XPOS":
                    word.tag_ = value
                elif key == "FEATS":
                    word.morph = spacy.tokens.morphanalysis.MorphAnalysis(
                        tok.vocab,
                        dict(feat.split("=") for feat in value.split("|")),
                    )
                elif key == "HEAD":
                    if value != "0":
                        word.head = doc[id_to_word[value]]
                elif key == "DEPREL":
                    word.dep_ = value
                else:
                    warnings.warn(f"Unused key {key} in CoNLL dict, ignoring it.")

        return doc


@registry.factory.register("eds.omop_dict2doc", spacy_compatible=False)
class OmopDict2DocConverter:
    """
    Examples
    --------

    ```{ .python .no-check }
    # Any kind of reader (`edsnlp.data.read/from_...`) can be used here
    docs = edsnlp.data.from_pandas(
        df,
        converter="omop",

        # Optional parameters
        tokenizer=tokenizer,
        doc_attributes=["note_datetime"],

        # Parameters below should only matter if you plan to import entities
        # from the dataframe. If the data doesn't contain pre-annotated
        # entities, you can ignore these.
        span_setter={"ents": True, "*": True},
        span_attributes={"negation": "negated"},
        default_attributes={"negated": False, "temporality": "present"},
    )
    ```

    Parameters
    ----------
    nlp: Optional[PipelineProtocol]
        The pipeline object (optional and likely not needed, prefer to use the
        `tokenizer` directly argument instead).
    tokenizer: Optional[Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_pipeline` in a
          [Stream][edsnlp.core.stream.Stream].
        - or the `eds` tokenizer by default.
    span_setter: SpanSetterArg
        The span setter to use when setting the spans in the documents. Defaults to
        setting the spans in the `ents` attribute, and creates a new span group for
        each JSON entity label.
    doc_attributes: AttributesMappingArg
        Mapping from JSON attributes to additional Span extensions (can be a list too).
        By default, all attributes are imported as Doc extensions with the same name.
    span_attributes: Optional[AttributesMappingArg]
        Mapping from JSON attributes to Span extensions (can be a list too).
        By default, all attributes are imported as Span extensions with the same name.
    default_attributes: AttributesMappingArg
        How to set attributes on spans for which no attribute value was found in the
        input format. This is especially useful for negation, or frequent attributes
        values (e.g. "negated" is often False, "temporal" is often "present"), that
        annotators may not want to annotate every time.
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[Tokenizer] = None,
        span_setter: SpanSetterArg = {"ents": True, "*": True},
        doc_attributes: AttributesMappingArg = {"note_datetime": "note_datetime"},
        span_attributes: Optional[AttributesMappingArg] = None,
        default_attributes: AttributesMappingArg = {},
        bool_attributes: AsList[str] = [],
    ):
        self.tokenizer = tokenizer
        self.span_setter = span_setter
        self.doc_attributes = doc_attributes
        self.span_attributes = span_attributes
        self.default_attributes = default_attributes
        for attr in bool_attributes:
            self.default_attributes[attr] = False

    def __call__(self, obj, tokenizer=None):
        # tok = get_current_tokenizer() if self.tokenizer is None else self.tokenizer
        tok = tokenizer or self.tokenizer or get_current_tokenizer()
        doc = tok(obj["note_text"] or "")
        doc._.note_id = obj.get("note_id", obj.get(FILENAME))
        for obj_name, ext_name in self.doc_attributes.items():
            if not Doc.has_extension(ext_name):
                Doc.set_extension(ext_name, default=None)
            doc._.set(ext_name, obj.get(obj_name))

        spans = []

        for dst in (
            *(() if self.span_attributes is None else self.span_attributes.values()),
            *self.default_attributes,
        ):
            if not Span.has_extension(dst):
                Span.set_extension(dst, default=None)

        for ent in obj.get("entities") or ():
            ent = dict(ent)
            span = doc.char_span(
                ent.pop("start_char"),
                ent.pop("end_char"),
                label=ent.pop("note_nlp_source_value"),
                alignment_mode="expand",
            )
            for label, value in ent.items():
                new_name = (
                    self.span_attributes.get(label, None)
                    if self.span_attributes is not None
                    else label
                )
                if self.span_attributes is None and not Span.has_extension(new_name):
                    Span.set_extension(new_name, default=None)

                if new_name:
                    span._.set(new_name, value)
            spans.append(span)

        set_spans(doc, spans, span_setter=self.span_setter)
        for attr, value in self.default_attributes.items():
            for span in spans:
                if span._.get(attr) is None:
                    span._.set(attr, value)
        return doc


@registry.factory.register("eds.omop_doc2dict", spacy_compatible=False)
class OmopDoc2DictConverter:
    """
    Examples
    --------

    ```{ .python .no-check }
    # Any kind of writer (`edsnlp.data.write/to_...`) can be used here
    df = edsnlp.data.to_pandas(
        docs,
        converter="omop",

        # Optional parameters
        span_getter={"ents": True},
        doc_attributes=["note_datetime"],
        span_attributes=["negation", "family"],
    )
    # or docs.to_pandas(...) if it's already a
    # [Stream][edsnlp.core.stream.Stream]
    ```

    Parameters
    ----------
    span_getter: SpanGetterArg
        The span getter to use when getting the spans from the documents. Defaults to
        getting the spans in the `ents` attribute.
    doc_attributes: AttributesMappingArg
        Mapping from Doc extensions to JSON attributes (can be a list too).
        By default, no doc attribute is exported, except `note_id`.
    span_attributes: AttributesMappingArg
        Mapping from Span extensions to JSON attributes (can be a list too).
        By default, no attribute is exported.
    """

    def __init__(
        self,
        *,
        span_getter: SpanGetterArg = {"ents": True},
        doc_attributes: AttributesMappingArg = {},
        span_attributes: AttributesMappingArg = {},
    ):
        self.span_getter = span_getter
        self.doc_attributes = doc_attributes
        self.span_attributes = span_attributes

    def __call__(self, doc):
        spans = get_spans(doc, self.span_getter)
        span_binding_getters = {
            obj_name: BINDING_GETTERS[
                ("_." + ext_name)
                if ext_name.split(".")[0] not in SPAN_BUILTIN_ATTRS
                else ext_name
            ]
            for ext_name, obj_name in self.span_attributes.items()
        }
        obj = {
            FILENAME: doc._.note_id,
            "note_id": doc._.note_id,
            "note_text": doc.text,
            **{
                obj_name: getattr(doc._, ext_name)
                for ext_name, obj_name in self.doc_attributes.items()
                if doc._.has(ext_name)
            },
            "entities": [
                {
                    "note_nlp_id": i,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "lexical_variant": ent.text,
                    "note_nlp_source_value": ent.label_,
                    **{
                        obj_name: value
                        for obj_name, value in (
                            (k, getter(ent))
                            for k, getter in span_binding_getters.items()
                        )
                    },
                }
                for i, ent in enumerate(sorted(dict.fromkeys(spans)))
            ],
        }
        return obj


@registry.factory.register("eds.ents_doc2dict", spacy_compatible=False)
class EntsDoc2DictConverter:
    """
    Parameters
    ----------
    span_getter: SpanGetterArg
        The span getter to use when getting the spans from the documents. Defaults to
        getting the spans in the `ents` attribute.
    doc_attributes: AttributesMappingArg
        Mapping from Doc extensions to JSON attributes (can be a list too).
        By default, no doc attribute is exported, except `note_id`.
    span_attributes: AttributesMappingArg
        Mapping from Span extensions to JSON attributes (can be a list too).
        By default, no attribute is exported.
    """

    def __init__(
        self,
        *,
        span_getter: SpanGetterArg = {"ents": True},
        doc_attributes: AttributesMappingArg = {},
        span_attributes: AttributesMappingArg = {},
    ):
        self.span_getter = span_getter
        self.doc_attributes = doc_attributes
        self.span_attributes = span_attributes

    def __call__(self, doc):
        span_binding_getters = {
            obj_name: BINDING_GETTERS[
                ("_." + ext_name)
                if ext_name.split(".")[0] not in SPAN_BUILTIN_ATTRS
                else ext_name
            ]
            for ext_name, obj_name in self.span_attributes.items()
        }
        doc_attributes_values = {
            obj_name: BINDING_GETTERS["_." + ext_name](doc)
            for ext_name, obj_name in self.doc_attributes.items()
        }
        return [
            {
                "note_id": doc._.note_id,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "lexical_variant": ent.text,
                "span_type": group,  # for backward compatibility
                **{
                    obj_name: getter(ent)
                    for obj_name, getter in span_binding_getters.items()
                },
                **doc_attributes_values,
            }
            for ent, group in sorted(
                dict(get_spans_with_group(doc, self.span_getter)).items()
            )
        ]


def get_dict2doc_converter(
    converter: Union[str, Callable], kwargs
) -> Tuple[Callable, Dict]:
    if not callable(converter):
        available = edsnlp.registry.factory.get_available()
        try:
            filtered = [
                name
                for name in available
                if converter == name or (converter in name and "dict2doc" in name)
            ]
            converter = edsnlp.registry.factory.get(filtered[0])
            nlp = kwargs.pop("nlp", None)
            if nlp is not None and "tokenizer" not in kwargs:
                kwargs["tokenizer"] = nlp.tokenizer
            converter = converter(**kwargs)
            kwargs = {}
            return converter, kwargs
        except (KeyError, IndexError):
            available = [v for v in available if "dict2doc" in v]
            raise ValueError(
                f"Cannot find converter for format {converter}. "
                f"Available converters are {', '.join(available)}"
            )
    if isinstance(converter, type):
        return converter(**kwargs), {}
    return converter, validate_kwargs(converter, kwargs)


def get_doc2dict_converter(
    converter: Union[str, Callable], kwargs
) -> Tuple[Callable, Dict]:
    if not callable(converter):
        available = edsnlp.registry.factory.get_available()
        try:
            filtered = [
                name
                for name in available
                if converter == name or (converter in name and "doc2dict" in name)
            ]
            converter = edsnlp.registry.factory.get(filtered[0])
            converter = converter(**kwargs)
            kwargs = {}
            return converter, kwargs
        except (KeyError, IndexError):
            available = [v for v in available if "doc2dict" in v]
            raise ValueError(
                f"Cannot find converter for format {converter}. "
                f"Available converters are {', '.join(available)}"
            )
    return converter, validate_kwargs(converter, kwargs)
