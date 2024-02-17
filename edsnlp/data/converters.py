"""
Converters are used to convert documents between python dictionaries and Doc objects.
There are two types of converters: readers and writers. Readers convert dictionaries to
Doc objects, and writers convert Doc objects to dictionaries.
"""
import contextlib
import inspect
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

from confit.registry import ValidatedFunction
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Span

import edsnlp
from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.utils.bindings import BINDING_GETTERS
from edsnlp.utils.span_getters import (
    SpanGetterArg,
    SpanSetterArg,
    get_spans,
    get_spans_with_group,
    set_spans,
)

FILENAME = "__FILENAME__"

SCHEMA = {}

_DEFAULT_TOKENIZER = None


def without_filename(d):
    d.pop(FILENAME, None)
    return d


def validate_kwargs(converter, kwargs):
    converter: FunctionType = copy(converter)
    spec = inspect.getfullargspec(converter)
    first = spec.args[0]
    converter.__annotations__[first] = Optional[Any]
    converter.__defaults__ = (None, *(spec.defaults or ())[-len(spec.args) + 1 :])
    vd = ValidatedFunction(converter, {"arbitrary_types_allowed": True})
    model = vd.init_model_instance(**kwargs)
    d = {
        k: v
        for k, v in model._iter()
        if (k in model.__fields__ or model.__fields__[k].default_factory)
    }
    d.pop("v__duplicate_kwargs", None)  # see pydantic ValidatedFunction code
    d.pop(vd.v_args_name, None)
    d.pop(first, None)
    return {**(d.pop(vd.v_kwargs_name, None) or {}), **d}


class SequenceStr:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, config=None) -> Sequence[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        raise ValueError("Not a string or list of strings")


if TYPE_CHECKING:
    SequenceStr = Union[str, Sequence[str]]  # noqa: F811


class AttributesMappingArg:
    """
    A mapping from JSON attributes to Span extensions (can be a list too).

    For instance:

    - `doc_attributes={"datetime": "note_datetime"}` will map the `datetime` JSON
      attribute to the `note_datetime` extension.
    - `doc_attributes="note_datetime"` will map the `note_datetime` JSON attribute to
      the `note_datetime` extension.
    - `span_attributes=["negation", "family"]` will map the `negation` and `family` JSON
      attributes to the `negation` and `family` extensions.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

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
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = edsnlp.blank("eds").tokenizer
    return _DEFAULT_TOKENIZER


@contextlib.contextmanager
def set_current_tokenizer(tokenizer):
    global _DEFAULT_TOKENIZER
    old = _DEFAULT_TOKENIZER
    if tokenizer:
        _DEFAULT_TOKENIZER = tokenizer
    yield
    _DEFAULT_TOKENIZER = old


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
    tokenizer: Optional[spacy.tokenizer.Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_pipeline` in a
          [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
        - or the `eds` tokenizer by default.
    span_setter: SpanSetterArg
        The span setter to use when setting the spans in the documents. Defaults to
        setting the spans in the `ents` attribute, and creates a new span group for
        each JSON entity label.
    span_attributes: Optional[AttributesMappingArg]
        Mapping from JSON attributes to Span extensions (can be a list too).
        By default, all attributes are imported as Span extensions with the same name.
    keep_raw_attribute_values: bool
        Whether to keep the raw attribute values (as strings) or to convert them to
        Python objects (e.g. booleans).
    default_attributes: AttributesMappingArg
        How to set attributes on spans for which no attribute value was found in the
        input format. This is especially useful for negation, or frequent attributes
        values (e.g. "negated" is often False, "temporal" is often "present"), that
        annotators may not want to annotate every time.
    """

    def __init__(
        self,
        nlp: Optional[PipelineProtocol] = None,
        *,
        tokenizer: Optional[Tokenizer] = None,
        span_setter: SpanSetterArg = {"ents": True, "*": True},
        span_attributes: Optional[AttributesMappingArg] = None,
        keep_raw_attribute_values: bool = False,
        bool_attributes: SequenceStr = [],
        default_attributes: AttributesMappingArg = {},
    ):
        self.tokenizer = tokenizer or (nlp.tokenizer if nlp is not None else None)
        self.span_setter = span_setter
        self.span_attributes = span_attributes  # type: ignore
        self.keep_raw_attribute_values = keep_raw_attribute_values
        self.default_attributes = default_attributes
        for attr in bool_attributes:
            self.default_attributes[attr] = False

    def __call__(self, obj):
        tok = get_current_tokenizer() if self.tokenizer is None else self.tokenizer
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
            for fragment in ent["fragments"]:
                span = doc.char_span(
                    fragment["begin"],
                    fragment["end"],
                    label=ent["label"],
                    alignment_mode="expand",
                )
                for label, value in ent["attributes"].items():
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
    # Any kind of reader (`edsnlp.data.read/from_...`) can be used here
    docs = edsnlp.data.write_standoff(
        "path/to/standoff",
        converter="standoff",  # set by default

        # Optional parameters
        span_getter={"ents": True},
        span_attributes={"negation": "negated"},
    )
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
                        obj_name: getattr(ent._, ext_name)
                        for ext_name, obj_name in self.span_attributes.items()
                        if ent._.has(ext_name)
                    },
                    "label": ent.label_,
                }
                for i, ent in enumerate(sorted(dict.fromkeys(spans)))
            ],
        }
        return obj


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
    tokenizer: Optional[spacy.tokenizer.Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_pipeline` in a
          [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
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
        nlp: Optional[PipelineProtocol] = None,
        *,
        tokenizer: Optional[PipelineProtocol] = None,
        span_setter: SpanSetterArg = {"ents": True, "*": True},
        doc_attributes: AttributesMappingArg = {},
        span_attributes: Optional[AttributesMappingArg] = None,
        default_attributes: AttributesMappingArg = {},
        bool_attributes: SequenceStr = [],
    ):
        self.tokenizer = tokenizer or (nlp.tokenizer if nlp is not None else None)
        self.span_setter = span_setter
        self.doc_attributes = doc_attributes
        self.span_attributes = span_attributes
        self.default_attributes = default_attributes
        for attr in bool_attributes:
            self.default_attributes[attr] = False

    def __call__(self, obj):
        tok = get_current_tokenizer() if self.tokenizer is None else self.tokenizer
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
    # Any kind of reader (`edsnlp.data.read/from_...`) can be used here
    docs = edsnlp.data.to_pandas(
        docs,
        converter="omop",

        # Optional parameters
        span_getter={"ents": True},
        doc_attributes=["note_datetime"],
        span_attributes=["negation", "family"],
    )
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
                        obj_name: getattr(ent._, ext_name)
                        for ext_name, obj_name in self.span_attributes.items()
                        if ent._.has(ext_name)
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
            obj_name: BINDING_GETTERS["_." + ext_name]
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
    kwargs_to_init = False
    if not callable(converter):
        available = edsnlp.registry.factory.get_available()
        try:
            filtered = [
                name
                for name in available
                if converter == name or (converter in name and "dict2doc" in name)
            ]
            converter = edsnlp.registry.factory.get(filtered[0])
            converter = converter(**kwargs).instantiate(nlp=None)
            kwargs = {}
            return converter, kwargs
        except (KeyError, IndexError):
            available = [v for v in available if "dict2doc" in v]
            raise ValueError(
                f"Cannot find converter for format {converter}. "
                f"Available converters are {', '.join(available)}"
            )
    if isinstance(converter, type) or kwargs_to_init:
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
            converter = converter(**kwargs).instantiate(nlp=None)
            kwargs = {}
            return converter, kwargs
        except (KeyError, IndexError):
            available = [v for v in available if "doc2dict" in v]
            raise ValueError(
                f"Cannot find converter for format {converter}. "
                f"Available converters are {', '.join(available)}"
            )
    return converter, validate_kwargs(converter, kwargs)
