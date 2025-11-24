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

import spacy
from confit.registry import ValidatedFunction
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Span
from typing_extensions import Literal

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
        fields = vd.model.model_fields
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


# ex: `[The [cat](ANIMAL) is [black](COLOR hex="#000000")].


@registry.factory.register("eds.markup_to_doc", spacy_compatible=False)
class MarkupToDocConverter:
    """
    Examples
    --------
    ```python
    import edsnlp

    # Any kind of reader (`edsnlp.data.read/from_...`) can be used here
    # If input items are dicts, the converter expects a "text" key/column.
    docs = list(
        edsnlp.data.from_iterable(
            [
                "This [is](VERB negation=True) not a [test](NOUN).",
                "This is another [test](NOUN).",
            ],
            converter="markup",
            span_setter="entities",
        ),
    )
    print(docs[0].spans["entities"])
    # Out: [is, test]
    ```

    You can also use it directly on a string:

    ```python
    from edsnlp.data.converters import MarkupToDocConverter

    converter = MarkupToDocConverter(
        span_setter={"verb": "VERB", "noun": "NOUN"},
        preset="xml",
    )
    doc = converter("This <VERB negation=True>is</VERB> not a <NOUN>test</NOUN>.")
    print(doc.spans["verb"])
    # Out: [is]
    print(doc.spans["verb"][0]._.negation)
    # Out: True
    ```

    Parameters
    ----------
    preset: Literal["md", "xml"]
        The preset to use for the markup format. Defaults to "md" (Markdown-like
        syntax). Use "xml" for XML-like syntax.
    opener: Optional[str]
        The regex pattern to match the opening tag of the markup. Defaults to the
        preset's opener.
    closer: Optional[str]
        The regex pattern to match the closing tag of the markup. Defaults to the
        preset's closer.
    tokenizer: Optional[Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_pipeline` in a
          [Stream][edsnlp.core.stream.Stream].
        - or the `eds` tokenizer by default.
    span_setter: SpanSetterArg
        The span setter to use when setting the spans in the documents. Defaults to
        setting the spans in the `ents` attribute and creates a new span group for
        each JSON entity label.
    span_attributes: Optional[AttributesMappingArg]
        Mapping from markup attributes to Span extensions (can be a list too).
        By default, all attributes are imported as Span extensions with the same name.
    keep_raw_attribute_values: bool
        Whether to keep the raw attribute values (as strings) or to convert them to
        Python objects (e.g. booleans).
    default_attributes: AttributesMappingArg
        How to set attributes on spans for which no attribute value was found in the
        input format. This is especially useful for negation, or frequent attributes
        values (e.g. "negated" is often False, "temporal" is often "present"), that
        annotators may not want to annotate every time.
    bool_attributes: AsList[str]
        List of boolean attributes to set to False by default. This is useful for
        attributes that are often not annotated, but you want to have a default value
        for them.
    """

    PRESETS = {
        "md": {
            # "["
            "opener": r"(?P<opener>\[)",
            # "](LABEL attr1=val1 attr2='val 2')"
            "closer": r"(?P<closer>\]\(\s*(?P<closer_label>[^\s\)\]\[\(<>\/]+)\s*(?P<closer_attrs>.*?)\))",  # noqa: E501
        },
        "xml": {
            # "<LABEL attr1=val1 attr2='val 2'>"
            "opener": r"(?P<opener><(?P<opener_label>[^\s\)\]\[\(<>\/]+)(?P<opener_attrs>.*?)>)",  # noqa: E501
            # "</LABEL>"
            "closer": r"(?P<closer></(?P<closer_label>[^\s\)\]\[\(<>\/]+)>)",
        },
    }

    def __init__(
        self,
        *,
        tokenizer: Optional[Tokenizer] = None,
        span_setter: SpanSetterArg = {"ents": True, "*": True},
        span_attributes: Optional[AttributesMappingArg] = None,
        keep_raw_attribute_values: bool = False,
        default_attributes: AttributesMappingArg = {},
        bool_attributes: AsList[str] = [],
        attr_splitter=r"\w+(?:='[^']*'|=\"[^\"]*\"|=[^\s]+)?",
        preset: Literal["md", "xml"] = "md",
        opener: Optional[str] = None,
        closer: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.span_setter = span_setter
        self.span_attributes = span_attributes
        self.keep_raw_attribute_values = keep_raw_attribute_values
        self.default_attributes = dict(default_attributes)
        for attr in bool_attributes:
            self.default_attributes[attr] = False
        self.opener = opener or self.PRESETS[preset]["opener"]
        self.closer = closer or self.PRESETS[preset]["closer"]
        self.attr_splitter = attr_splitter

    def _as_python(self, value: str):
        import ast

        if self.keep_raw_attribute_values:
            return value
        try:
            return ast.literal_eval(value)
        except Exception:
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
        return value

    def _parse(self, inline_text: str):
        import re

        last_inline_offset = 0
        pattern = f"{self.closer}|{self.opener}"
        seps = list(re.finditer(pattern, inline_text))
        pairs = []
        stack = []
        text = ""
        for sep in seps:
            text += inline_text[last_inline_offset : sep.start()]
            offset = len(text)
            last_inline_offset = sep.end()
            sep_gd = sep.groupdict()
            if sep_gd.get("opener"):
                stack.insert(0, (sep_gd, offset))
            elif sep_gd.get("closer"):
                label = sep_gd.get("closer_label")
                if stack:
                    matching_opener = next(
                        (
                            (op_gd, start)
                            for (op_gd, start) in stack
                            if "opener_label" not in op_gd
                            or op_gd.get("opener_label") == label
                            or not label
                        ),
                        None,
                    )
                    if matching_opener is not None:
                        stack.remove(matching_opener)
                        op_gd, start = matching_opener
                        pairs.append((start, offset, op_gd, sep_gd))
        text += inline_text[last_inline_offset:]
        pairs = sorted(pairs, key=lambda x: (x[0], -x[1]))

        entities = []
        for i, (start, end, opener_gd, closer_gd) in enumerate(pairs):
            label = (
                opener_gd.get("opener_label")
                if opener_gd.get("opener_label")
                else closer_gd.get("closer_label")
            )
            attrs_str = " ".join(
                (
                    (opener_gd.get("opener_attrs", "")),
                    (closer_gd.get("closer_attrs", "")),
                )
            )
            attrs = {}
            for attr in re.findall(self.attr_splitter, attrs_str.strip()):
                if not attr:
                    continue
                if "=" in attr:
                    k, v = attr.split("=", 1)
                    attrs[k] = self._as_python(v)
                else:
                    attrs[attr] = True
            entities.append((start, end, label, attrs))
        entities = sorted(entities)
        return text, entities

    def __call__(self, obj, tokenizer=None):
        tok = tokenizer or self.tokenizer or get_current_tokenizer()
        if isinstance(obj, str):
            obj = {"text": obj}
        annotated = obj["text"]
        plain, raw_ents = self._parse(annotated)

        doc = tok(plain)
        doc._.note_id = obj.get("doc_id", obj.get(FILENAME))

        for dst in (
            *(() if self.span_attributes is None else self.span_attributes.values()),
            *self.default_attributes,
        ):
            if not Span.has_extension(dst):
                Span.set_extension(dst, default=None)

        spans = []
        for start, end, label, attrs in raw_ents:
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if span is None:
                continue
            for k, v in attrs.items():
                new_k = (
                    self.span_attributes.get(k)
                    if self.span_attributes is not None
                    else k
                )
                if self.span_attributes is None and not Span.has_extension(new_k):
                    Span.set_extension(new_k, default=None)
                if new_k:
                    span._.set(new_k, v)
            spans.append(span)

        set_spans(doc, spans, span_setter=self.span_setter)
        for attr, value in self.default_attributes.items():
            for span in spans:
                if span._.get(attr) is None:
                    span._.set(attr, value)

        return doc


@registry.factory.register("eds.doc_to_markup", spacy_compatible=False)
class DocToMarkupConverter:
    """
    Convert a Doc to a string with inline markup.

    This is the inverse of :class:`MarkupToDocConverter`. It renders selected
    spans as either Markdown-like tags (``[text](LABEL key=val ...)``)
    or XML-like tags (``<LABEL key=val ...>text</LABEL>``).

    Parameters
    ----------
    span_getter : SpanGetterArg, default {"ents": True}
        Which spans to render from the document.
    span_attributes : AttributesMappingArg, default {}
        Mapping from Span extensions (or builtins like ``label_``, ``kb_id_``)
        to attribute names in the rendered markup. Only attributes with a
        non-``None`` value are emitted.
    default_attributes : AttributesMappingArg, default {}
        When an attribute equals its provided default value, it is omitted
        from the output (e.g., avoid printing ``negated=False`` when ``False``
        is the default).
    preset : Literal["md", "xml"], default "md"
        Output syntax. ``"md"`` produces the Markdown‑like form, ``"xml"`` the
        XML‑like form.
    """

    def __init__(
        self,
        *,
        span_getter: SpanGetterArg = {"ents": True},
        span_attributes: AttributesMappingArg = {},
        default_attributes: AttributesMappingArg = {},
        bool_attributes: AsList[str] = [],
        preset: Literal["md", "xml"] = "md",
    ):
        self.span_getter = span_getter
        self.span_attributes = span_attributes
        self.default_attributes = dict(default_attributes)
        self.bool_attributes = bool_attributes
        for attr in bool_attributes:
            self.default_attributes[attr] = False
        self.preset = preset

    def _format_attr(self, key: str, value: Any) -> str:
        if key in self.bool_attributes:
            return f"{key}" if value else ""

        if isinstance(value, (bool, int, float)):
            return repr(value)
        s = str(value)
        # Quote strings containing spaces or tag delimiters so they round‑trip
        if any(c.isspace() for c in s) or any(c in "<>[]()" for c in s):
            s = repr(s)
        return f"{key}={s}"

    def __call__(self, doc: Doc) -> str:
        # Build getters once
        span_binding_getters = {
            obj_name: BINDING_GETTERS[
                ("_." + ext_name)
                if ext_name.split(".")[0] not in SPAN_BUILTIN_ATTRS
                else ext_name
            ]
            for ext_name, obj_name in (self.span_attributes or {}).items()
        }

        # Collect and dedupe spans
        spans = list(sorted(dict.fromkeys(get_spans(doc, self.span_getter))))

        text = doc.text
        starts: Dict[int, list[Span]] = {}
        ends: Dict[int, list[Span]] = {}
        for sp in spans:
            attrs = {
                obj_name: getter(sp)
                for obj_name, getter in span_binding_getters.items()
            }
            attrs = {
                k: v
                for k, v in attrs.items()
                if v is not None
                and not (
                    k in self.default_attributes and self.default_attributes[k] == v
                )
            }
            attrs_str = (
                " ".join(self._format_attr(k, v) for k, v in attrs.items())
                if attrs
                else ""
            )
            starts.setdefault(sp.start_char, []).append((sp, attrs_str))
            ends.setdefault(sp.end_char, []).append((sp, attrs_str))

        out: list[str] = []
        last = 0

        positions = sorted({*starts.keys(), *[sp.end_char for sp in spans]})
        for pos in positions:
            pos = min(pos, len(text))
            if pos > last:
                out.append(text[last:pos])

            # Close any spans that end here (LIFO to preserve nesting)
            for sp, attrs_str in sorted(
                ends.get(pos, []),
                key=lambda s: s[0].start_char,
                reverse=True,
            ):
                if self.preset == "md":
                    out.append(
                        f"]({sp.label_}{(' ' + attrs_str) if attrs_str else ''})"
                    )
                else:  # xml
                    out.append(f"</{sp.label_}>")

            # Open any spans that start here (FIFO to preserve nesting)
            for sp, attrs_str in sorted(
                starts.get(pos, []),
                key=lambda s: s[0].end_char,
            ):
                if self.preset == "md":
                    out.append("[")
                else:  # xml
                    out.append(f"<{sp.label_}{(' ' + attrs_str) if attrs_str else ''}>")

            last = pos

        # Trailing text after the last event
        if last < len(text):
            out.append(text[last:])

        return "".join(out)


def get_dict2doc_converter(
    converter: Union[str, Callable], kwargs
) -> Tuple[Callable, Dict]:
    if not callable(converter):
        available = edsnlp.registry.factory.get_available()
        try:
            filtered = [
                name
                for name in available
                if converter == name
                or (
                    converter in name
                    and (name.endswith("2doc") or name.endswith("to_doc"))
                )
            ]
            converter = edsnlp.registry.factory.get(filtered[0])
            nlp = kwargs.pop("nlp", None)
            if nlp is not None and "tokenizer" not in kwargs:
                kwargs["tokenizer"] = nlp.tokenizer
            converter = converter(**kwargs)
            kwargs = {}
            return converter, kwargs
        except (KeyError, IndexError):
            available = [
                v for v in available if (v.endswith("2doc") or v.endswith("to_doc"))
            ]
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
                if converter == name
                or (converter in name and ("doc2" in name or "doc_to" in name))
            ]
            converter = edsnlp.registry.factory.get(filtered[0])
            converter = converter(**kwargs)
            kwargs = {}
            return converter, kwargs
        except (KeyError, IndexError):
            available = [v for v in available if ("doc2" in v or "doc_to" in v)]
            raise ValueError(
                f"Cannot find converter for format {converter}. "
                f"Available converters are {', '.join(available)}"
            )
    return converter, validate_kwargs(converter, kwargs)
