import edsnlp
from edsnlp.pipes.qualifiers.llm.llm_qualifier import LLMSpanClassifier
from edsnlp.utils.examples import parse_example
from edsnlp.utils.span_getters import make_span_context_getter


def test_llm_span_classifier_basic():
    # Patch AsyncLLM to avoid real API calls
    class DummyAsyncLLM:
        def __init__(self, *args, **kwargs):
            pass

        async def __call__(self, batch_messages):
            # Return a dummy label for each message
            return ["True" for _ in batch_messages]

    import edsnlp.pipes.qualifiers.llm.llm_qualifier as llm_mod

    llm_mod.AsyncLLM = DummyAsyncLLM

    nlp = edsnlp.blank("eds")
    # nlp.add_pipe("sentencizer")
    # nlp.add_pipe(eds.dates(as_ents=True))

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
            system_prompt="You are a medical assistant",
            user_prompt=None,
            prefix_prompt="Is '{span}' a date? ",
            suffix_prompt="",
            examples=[
                (
                    "\nIs'07/12/2020' a date. 07/12/2020 : Anapath / biopsies rectales.",  # noqa: E501
                    "False",
                )
            ],
            api_url="http://dummy",
            max_tokens=10,
            response_mapping=None,
            extra_body=None,
            temperature=0.0,
            response_format=None,
            n_concurrent_tasks=1,
        )
    )
    doc = nlp(doc)

    # Check that the extension is set and the dummy label is applied
    for span in doc.ents:
        assert hasattr(span._, "test_attr")
        assert span._.test_attr == "True"
