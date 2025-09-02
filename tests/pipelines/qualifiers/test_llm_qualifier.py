from pytest import mark

import edsnlp
from edsnlp.pipes.qualifiers.llm.llm_qualifier import LLMSpanClassifier
from edsnlp.utils.examples import parse_example
from edsnlp.utils.span_getters import make_span_context_getter


@mark.parametrize("label", ["True", None])
@mark.parametrize("response_mapping", [{"^True$": "1", "^False$": "0"}, None])
def test_llm_span_classifier(label, response_mapping):
    # Patch AsyncLLM to avoid real API calls
    class DummyAsyncLLM:
        def __init__(self, *args, **kwargs):
            # Initialize the dummy LLM
            pass

        async def __call__(self, batch_messages):
            # Return a dummy label for each message
            return [label for _ in batch_messages]

    import edsnlp.pipes.qualifiers.llm.llm_qualifier as llm_mod

    llm_mod.AsyncLLM = DummyAsyncLLM

    nlp = edsnlp.blank("eds")
    example = "En RCP du <ent>20/02/2025</ent>, patient classé cT3a N0 M0, haut risque. IRM multiparamétrique du <ent>10/02/2025</ent>."  # noqa: E501
    text, entities = parse_example(example)
    doc = nlp(text)
    doc.ents = [
        doc.char_span(ent.start_char, ent.end_char, label="date") for ent in entities
    ]

    # LLMSpanClassifier
    nlp.add_pipe(
        LLMSpanClassifier(
            nlp=nlp,
            name="llm",
            model="dummy",
            span_getter={"ents": True},
            attributes={"_.test_attr": True},
            context_getter=make_span_context_getter(
                context_sents=0,
                context_words=(5, 5),
            ),
            prompt={
                "system_prompt": "You are a medical assistant.",
                "user_prompt": "You should help us identify dates in the text.",
                "prefix_prompt": "Is '{span}' a date? The text is as follows:\n<<< ",
                "suffix_prompt": " >>>",
                "examples": [
                    (
                        "\nIs'07/12/2020' a date. The text is as follows:\n<<< 07/12/2020 : Anapath / biopsies rectales. >>>",  # noqa: E501
                        "False",
                    )
                ],
            },
            api_url="https://dummy",
            api_params={
                "max_tokens": 10,
                "temperature": 0.0,
                "response_format": None,
                "extra_body": None,
            },
            response_mapping=response_mapping,
            n_concurrent_tasks=1,
        )
    )
    doc = nlp(doc)

    # Check that the extension is set and the dummy label is applied
    for span in doc.ents:
        assert hasattr(span._, "test_attr")
        if response_mapping is not None:
            if label == "True":
                assert span._.test_attr == "1"
            elif label is None:
                assert span._.test_attr is None

        if response_mapping is None:
            if label == "True":
                assert span._.test_attr == label
            elif label is None:
                assert span._.test_attr is None

    assert nlp.get_pipe("llm").attributes == {"_.test_attr": True}


@mark.parametrize(
    "prefix_prompt,suffix_prompt",
    [("Is '{span}' a date? The text is as follows:\n<<< ", " >>>"), (None, None)],
)
def test_llm_span_classifier_preprocess(prefix_prompt, suffix_prompt):
    # Patch AsyncLLM to avoid real API calls
    class DummyAsyncLLM:
        def __init__(self, *args, **kwargs):
            # Initialize the dummy LLM
            pass

        async def __call__(self, batch_messages):
            # Return a dummy label for each message
            return ["True" for _ in batch_messages]

    import edsnlp.pipes.qualifiers.llm.llm_qualifier as llm_mod

    llm_mod.AsyncLLM = DummyAsyncLLM

    nlp = edsnlp.blank("eds")
    example = "En RCP du <ent>20/02/2025</ent>, patient classé cT3a N0 M0, haut risque. IRM multiparamétrique du <ent>10/02/2025</ent>."  # noqa: E501
    text, entities = parse_example(example)
    doc = nlp(text)
    doc.ents = [
        doc.char_span(ent.start_char, ent.end_char, label="date") for ent in entities
    ]

    system_prompt = "You are a medical assistant."
    user_prompt = "You should help us identify dates in the text."
    examples = [
        (
            "\nIs'07/12/2020' a date. The text is as follows:\n<<< 07/12/2020 : Anapath / biopsies rectales. >>>",  # noqa: E501
            "False",
        )
    ]

    # LLMSpanClassifier
    llm = LLMSpanClassifier(
        nlp=nlp,
        name="llm",
        model="dummy",
        span_getter={"ents": True},
        attributes={"_.test_attr": True},
        context_getter=make_span_context_getter(
            context_sents=0,
            context_words=(5, 5),
        ),
        prompt={
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "prefix_prompt": prefix_prompt,
            "suffix_prompt": suffix_prompt,
            "examples": examples,
        },
        api_url="https://dummy",
        api_params={
            "max_tokens": 10,
            "temperature": 0.0,
            "response_format": None,
            "extra_body": None,
        },
        response_mapping=None,
        n_concurrent_tasks=1,
    )

    inputs = llm.preprocess(doc)
    if (prefix_prompt is not None) and (suffix_prompt is not None):
        assert inputs["doc_batch_messages"][0] == [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "user",
                "content": examples[0][0],
            },
            {"role": "assistant", "content": examples[0][1]},
            {
                "role": "user",
                "content": "Is '20/02/2025' a date? The text is as follows:\n<<< En RCP du 20/02/2025, patient classé cT3 >>>",  # noqa: E501
            },
        ]
    else:
        assert inputs["doc_batch_messages"][0] == [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "user",
                "content": examples[0][0],
            },
            {"role": "assistant", "content": examples[0][1]},
            {
                "role": "user",
                "content": "En RCP du 20/02/2025, patient classé cT3",  # noqa: E501
            },
        ]
