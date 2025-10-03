import datetime
import json
import re

import pytest

# Also serves as a py37 skip since we don't install openai in py37 CI
pytest.importorskip("openai")


from typing import Optional

from mock_llm_service import mock_llm_service
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    WithJsonSchema,
    field_validator,
)
from typing_extensions import Annotated

import edsnlp
import edsnlp.pipes as eds

PROMPT = """\
Predict JSON attributes for the highlighted entity.
Return keys `negation` (bool) and `date` (YYYY-MM-DD string, optional).
"""


class QualifierSchema(BaseModel):
    negation: bool
    date: Optional[datetime.date] = None


def assert_response_schema(response_format):
    assert response_format["type"] == "json_schema"
    payload = response_format["json_schema"]
    assert payload["name"]
    schema = payload["schema"]
    assert schema.get("type") == "object"
    assert schema.get("additionalProperties") is False
    props = schema.get("properties", {})
    assert "negation" in props
    neg_type = props["negation"].get("type")
    assert neg_type in {"boolean", "bool"}
    assert "date" in props


def test_llm_span_qualifier_sets_attributes(xml2doc, doc2xml):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.llm_span_qualifier(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            prompt=PROMPT,
            output_schema=QualifierSchema,
            context_formatter=doc2xml,
        )
    )

    doc = xml2doc("Le patient n'a pas de <DIAG>tuberculose</DIAG>.")

    def responder(*, messages, response_format, **_):
        assert_response_schema(response_format)
        user_prompt = messages[-1]["content"]
        assert "<DIAG>tuberculose</DIAG>" in user_prompt
        return '{"negation": true, "date": "2024-01-02"}'

    with mock_llm_service(responder=responder):
        doc = nlp(doc)

    (ent,) = doc.ents
    assert ent._.negation is True
    assert ent._.date is not None


def test_llm_span_qualifier_async_multiple_spans(xml2doc, doc2xml):
    def prompt(context, examples):
        assert len(examples) == 0
        messages = []
        system_content = (
            "You are a span classifier.\n"
            "Answer with JSON using the keys: biopsy_procedure.\n"
            "Here are some examples:\n"
        )
        for ex_context, ex_json in examples:
            system_content += f"- User: {ex_context}\n"
            system_content += f"  Assistant: {ex_json}\n"
        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": context})
        return messages

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.llm_span_qualifier(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            prompt=prompt,
            output_schema=QualifierSchema,
            context_formatter=doc2xml,
            max_concurrent_requests=2,
            on_error="raise",
            context_getter="words[-2:2]",
        )
    )

    doc = xml2doc(
        "Le patient a une <DIAG>tuberculose</DIAG> et une <DIAG>pneumonie</DIAG>."
    )

    def responder(*, messages, response_format, **_):
        assert_response_schema(response_format)
        content = messages[-1]["content"]
        if content == "a une <DIAG>tuberculose</DIAG> et une ":
            return '{"negation": true}'
        assert content == "et une <DIAG>pneumonie</DIAG>."
        return '{"negation": false, "date": "2024-06-01"}'

    with mock_llm_service(responder=responder):
        doc = nlp(doc)

    values = {
        ent.text: (
            ent._.negation,
            ent._.date,
        )
        for ent in doc.ents
    }
    assert values == {
        "tuberculose": (True, None),
        "pneumonie": (False, datetime.date(2024, 6, 1)),
    }


def test_llm_span_qualifier_multi_text(xml2doc, doc2xml):
    nlp = edsnlp.blank("eds")

    example_docs = edsnlp.data.from_iterable(
        [
            "Le <PAT date=2012-01-01>patient</PAT> est atteint de <DIAG>covid</DIAG>.",
            (
                "Diagnostic de <DIAG negation=true date=2025-06-01>pneumonie</DIAG> "
                "non retenu le 1er juin 2025."
            ),
        ],
        converter="markup",
        preset="xml",
        bool_attributes=["negation"],
    )

    nlp.add_pipe(
        eds.llm_span_qualifier(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            prompt=PROMPT,
            output_schema=QualifierSchema,
            context_formatter=doc2xml,
            max_concurrent_requests=3,
            on_error="warn",
            attributes={"_.negation": ["DIAG", "PAT"], "date": ["DIAG"]},
            examples=example_docs,
            use_retriever=True,
            max_few_shot_examples=2,
        )
    )

    docs = [
        xml2doc("Le <PAT>patient</PAT> n'a pas la <DIAG>covid</DIAG> le 10/11/12."),
        xml2doc("Une <DIAG>pneumonie</DIAG> a été diagnostiquée le 15 juin 2024."),
        xml2doc("On suspecte une <DIAG>grippe</DIAG>."),
    ]

    def responder(*, messages, response_format, **_):
        assert_response_schema(response_format)
        content = messages[-1]["content"]
        if "<DIAG>covid</DIAG>" in content:
            return '{"negation": true, "date": "2012-11-10"}'
        if "<DIAG>pneumonie</DIAG>" in content:
            return '{"negation": false, "date": "2024-06-15"}'
        if "<PAT>patient</PAT>" in content:
            return '{"negation": false, "date": null}'
        assert "<DIAG>grippe</DIAG>" in content
        return '```json\n{"negation": false}```'

    with mock_llm_service(responder=responder):
        processed = list(nlp.pipe(docs))

    results = [
        [(ent.text, ent._.negation, ent._.date) for ent in doc.ents]
        for doc in processed
    ]
    assert results == [
        [("patient", False, None), ("covid", True, datetime.date(2012, 11, 10))],
        [("pneumonie", False, datetime.date(2024, 6, 15))],
        [("grippe", False, None)],
    ]


def test_llm_span_qualifier_async_error(xml2doc, doc2xml):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.llm_span_qualifier(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            prompt=PROMPT,
            output_schema=QualifierSchema,
            context_formatter=doc2xml,
            max_concurrent_requests=2,
            on_error="warn",
        )
    )

    doc = xml2doc(
        "Le patient a une <DIAG>tuberculose</DIAG> et une <DIAG>pneumonie</DIAG>."
    )

    def responder(*, response_format, **_):
        assert_response_schema(response_format)
        raise ValueError("Simulated error")

    with mock_llm_service(responder=responder), pytest.warns(
        UserWarning, match="request failed"
    ):
        doc = nlp(doc)

    for ent in doc.ents:
        assert ent._.negation is None
        assert ent._.date is None


def test_yes_no_schema(xml2doc, doc2xml):
    YesBool = Annotated[
        bool,
        WithJsonSchema({"type": "string", "enum": ["yes", "no"]}),
        BeforeValidator(lambda v: v.lower() in {"yes", "y", "true", "1"}),
        PlainSerializer(lambda v: "yes" if v else "no", when_used="json"),
    ]

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.llm_span_qualifier(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            prompt="""\
Determine if the highlighted entity is negated.
Answer "yes" or "no".""",
            output_schema=YesBool,
            attributes="negation",
            context_formatter=doc2xml,
            on_error="warn",
            examples=edsnlp.data.from_iterable(
                [
                    "Le patient a eu une <DIAG>pneumonie</DIAG> le 1er mai 2024.",
                    "Le patient n'a pas le <DIAG negation=true>covid</DIAG>.",
                ],
                converter="markup",
                preset="xml",
                bool_attributes=["negation"],
            ),
        )
    )

    doc = xml2doc("Le patient n'a pas de <DIAG>tuberculose</DIAG>.")

    def responder(*, messages, **_):
        user_prompt = messages[-1]["content"]
        # Ideally, LLM service should support scalar schemas
        # but this isn't the case yet.

        # assert response_format["type"] == "json_schema"
        # payload = response_format["json_schema"]
        # assert payload["name"]
        # schema = payload["schema"]
        # assert schema.get("type") == "string"
        # assert schema.get("enum") == ["yes", "no"]

        assert "<DIAG>tuberculose</DIAG>" in user_prompt
        return "yes"

    with mock_llm_service(responder=responder):
        doc = nlp(doc)

    (ent,) = doc.ents
    assert ent._.negation is True
    assert ent._.date is None


def test_empty_schema(xml2doc, doc2xml):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.llm_span_qualifier(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            prompt="""\
For the highlighted entity, determine the type of the entity as a single phrase.
""",
            attributes="type",
            context_formatter=doc2xml,
            examples=edsnlp.data.from_iterable(
                [
                    "On prescrit du <ENT>paracétamol</ENT>.",
                    "Le patient n'a pas le <ENT type=disease>covid</ENT>.",
                ],
                converter="markup",
                preset="xml",
            ),
        ),
    )
    doc = xml2doc("Le patient n'a pas de <ENT>tuberculose</ENT>.")

    def responder(*, messages, **_):
        user_prompt = messages[-1]["content"]
        assert "<ENT>tuberculose</ENT>" in user_prompt
        return "disease"

    with mock_llm_service(responder=responder):
        doc = nlp(doc)
    (ent,) = doc.ents
    assert ent._.type == "disease"


def make_prompt_builder(system_prompt):
    def prompt(context, retrieved_examples):
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for ctx, answer in retrieved_examples:
            messages.append({"role": "user", "content": ctx})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": context})
        return messages

    return prompt


def make_context_formatter(prefix_prompt, suffix_prompt):
    def formatter(doc):
        (span,) = doc.ents
        context_text = doc.text.strip()
        if prefix_prompt is None or suffix_prompt is None:
            return context_text
        prefix = prefix_prompt.format(span=span.text)
        suffix = suffix_prompt.format(span=span.text)
        return f"{prefix}{context_text}{suffix}"

    return formatter


@pytest.mark.parametrize("label", ["True", None])
@pytest.mark.parametrize("response_mapping", [{"^True$": "1", "^False$": "0"}, None])
def test_llm_span_qualifier_custom_formatter_sets_attributes(
    xml2doc, label, response_mapping
):
    system_prompt = (
        "You are a medical assistant, build to help identify dates in the text."
    )
    prefix_prompt = "Is '{span}' a date? The text is as follows:\n<<< "
    suffix_prompt = " >>>"
    example_doc = xml2doc(
        "<DATE test_attr='False'>07/12/2020</DATE> : Anapath / biopsies rectales."  # noqa: E501
    )

    class ResponseSchema(BaseModel):
        test_attr: Optional[str] = None

        @field_validator("test_attr", mode="before")
        def apply_mapping(cls, value):
            if value is None:
                return None
            value_str = str(value)
            if response_mapping is None:
                return value_str
            for pattern, mapped in response_mapping.items():
                if re.match(pattern, value_str):
                    return mapped
            return value_str

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.llm_span_qualifier(
            api_url="http://localhost:8080/v1",
            model="dummy",
            name="llm",
            prompt=make_prompt_builder(system_prompt),
            output_schema=ResponseSchema,
            context_getter="words[-5:5]",
            context_formatter=make_context_formatter(prefix_prompt, suffix_prompt),
            max_concurrent_requests=1,
            max_few_shot_examples=1,
            examples=[example_doc],
            api_kwargs={
                "max_tokens": 10,
                "temperature": 0.0,
                "response_format": None,
                "extra_body": None,
            },
        )
    )

    doc = xml2doc(
        "En RCP du <DATE>20/02/2025</DATE>, patient classé cT3a N0 M0, haut risque. IRM multiparamétrique du <DATE>10/02/2025</DATE>."  # noqa: E501
    )
    qualifier = nlp.get_pipe("llm")
    retrieved_examples = qualifier.examples[:1]
    assert len(retrieved_examples) == 1
    example_context, example_answer = retrieved_examples[0]
    assert (
        example_context
        == "Is '07/12/2020' a date? The text is as follows:\n<<< 07/12/2020 : Anapath / biopsies rectales >>>"  # noqa: E501
    )
    context_doc = qualifier._build_context_doc(doc.ents[0])
    expected_context = qualifier.context_formatter(context_doc)
    expected_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example_context},
        {"role": "assistant", "content": example_answer},
        {"role": "user", "content": expected_context},
    ]
    assert qualifier.build_prompt(expected_context) == expected_messages

    def responder(*, messages, response_format=None, **_):
        if response_format is not None:
            assert_response_schema(response_format)
        payload = {"test_attr": label}
        return json.dumps(payload)

    with mock_llm_service(responder=responder):
        doc = nlp(doc)

    for span in doc.ents:
        assert hasattr(span._, "test_attr")
        if label == "True":
            if response_mapping is None:
                assert span._.test_attr == "True"
            else:
                assert span._.test_attr == "1"
        else:
            assert span._.test_attr is None

    assert qualifier.attributes == {"_.test_attr": True}


@pytest.mark.parametrize(
    ("prefix_prompt", "suffix_prompt"),
    [
        ("Is '{span}' a date? The text is as follows:\n<<< ", " >>>"),
        (None, None),
    ],
)
def test_llm_span_qualifier_custom_formatter_prompt(
    xml2doc, prefix_prompt, suffix_prompt
):
    system_prompt = (
        "You are a medical assistant, build to help identify dates in the text."
    )
    example_doc = xml2doc(
        "<DATE test_attr='False'>07/12/2020</DATE> : Anapath / biopsies rectales."  # noqa: E501
    )

    class ResponseSchema(BaseModel):
        test_attr: Optional[str] = None

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.llm_span_qualifier(
            api_url="http://localhost:8080/v1",
            model="dummy",
            name="llm",
            prompt=make_prompt_builder(system_prompt),
            output_schema=ResponseSchema,
            context_getter="words[-5:5]",
            context_formatter=make_context_formatter(prefix_prompt, suffix_prompt),
            max_concurrent_requests=1,
            max_few_shot_examples=1,
            examples=[example_doc],
            api_kwargs={
                "max_tokens": 10,
                "temperature": 0.0,
                "response_format": None,
                "extra_body": None,
            },
        )
    )

    doc = xml2doc(
        "En RCP du <DATE>20/02/2025</DATE>, patient classé cT3a N0 M0, haut risque. IRM multiparamétrique du <DATE>10/02/2025</DATE>."  # noqa: E501
    )

    qualifier = nlp.get_pipe("llm")
    retrieved_examples = qualifier.examples[:1]
    assert len(retrieved_examples) == 1
    example_context, example_answer = retrieved_examples[0]
    expected_example_context = (
        "Is '07/12/2020' a date? The text is as follows:\n<<< 07/12/2020 : Anapath / biopsies rectales >>>"  # noqa: E501
        if prefix_prompt is not None
        else "07/12/2020 : Anapath / biopsies rectales"
    )
    assert example_context == expected_example_context
    span = doc.ents[0]
    context_doc = qualifier._build_context_doc(span)
    context_text = qualifier.context_formatter(context_doc)
    expected_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example_context},
        {"role": "assistant", "content": example_answer},
        {"role": "user", "content": context_text},
    ]

    expected_context = (
        "Is '20/02/2025' a date? The text is as follows:\n<<< En RCP du 20/02/2025, patient classé cT3 >>>"  # noqa: E501
        if prefix_prompt is not None
        else "En RCP du 20/02/2025, patient classé cT3"
    )
    assert context_text == expected_context
    assert qualifier.build_prompt(context_text) == expected_messages

    def responder(*, messages, response_format=None, **_):
        if response_format is not None:
            assert_response_schema(response_format)
        return json.dumps({"test_attr": "True"})

    with mock_llm_service(responder=responder):
        doc = nlp(doc)

    span = doc.ents[0]
    assert span._.test_attr == "True"
    assert qualifier.attributes == {"_.test_attr": True}
