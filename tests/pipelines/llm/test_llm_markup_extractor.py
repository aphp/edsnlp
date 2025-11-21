import pytest

pytest.importorskip("openai")


from mock_llm_service import mock_llm_service

import edsnlp
import edsnlp.pipes as eds

PROMPT = """\
Extract entities using XML tags.
The tags to use are the following:
- <diagnosis>: A medical diagnosis
- <treatment>: A medical treatment
"""


def test_xml_markup_extractor(doc2md, md2doc):
    nlp = edsnlp.blank("eds")
    train_docs = [
        md2doc("Le patient a une [pneumonie](diagnosis)."),
        md2doc("On prescrit une [antibiothérapie](treatment)."),
    ]

    def prompt(doc_text, examples):
        assert len(examples) == 1
        system_content = (
            "You are a XML-based extraction assistant.\n"
            "Here are some examples of what's expected:\n"
        )
        for ex_text, ex_markup in examples:
            system_content += f"- User: {ex_text}\n"
            system_content += f"  Bot answer: {ex_markup}\n"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": doc_text},
        ]

    nlp.add_pipe(
        eds.llm_markup_extractor(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            examples=train_docs,
            prompt=prompt,
            use_retriever=True,
            max_few_shot_examples=1,
            max_concurrent_requests=5,
            on_error="raise",
        )
    )
    md = (
        "La patient souffre de [tuberculose](diagnosis) et "
        "on débute une [antibiothérapie](treatment)."
    )

    with mock_llm_service(
        responder=lambda **kw: (
            "La patiente souffre de <diagnosis>tuberculose</diagnosis> et "
            "on débute une <treatment>antibiothérapie</treatment>."
        )
    ):
        doc = md2doc(md)
        doc = nlp(doc.text)
        assert doc2md(doc) == md

    with mock_llm_service(
        responder=lambda **kw: ("une <diagnosis>pneumonie</diagnosis> du thorax"),
    ):
        md = "Le patient a une [pneumonie](diagnosis) du thorax, c'est très grave."
        doc = md2doc(md)
        doc = nlp(doc.text)
        assert doc2md(doc) == md


def test_xml_markup_extractor_multi_text(doc2md, md2doc):
    nlp = edsnlp.blank("eds")
    train_docs = [
        md2doc("Le patient a une [pneumonie](diagnosis)."),
        md2doc("On prescrit une [antibiothérapie](treatment)."),
    ]

    nlp.add_pipe(
        eds.llm_markup_extractor(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            examples=train_docs,
            prompt=PROMPT,
            use_retriever=True,
            max_few_shot_examples=1,
            max_concurrent_requests=5,
        )
    )
    md = [
        "La patient souffre de [tuberculose](diagnosis).",
        "On débute une [antibiothérapie](treatment) dès ajd.",
        "Il a une [pneumonie](diagnosis) du thorax.",
        "C'est très grave.",
    ]

    def responder(messages, **kw):
        assert len(messages) == 4  # 1 system + 1 few shot = (user, bot) + 1 user
        res = (
            messages[-1]["content"]
            .replace("tuberculose", "<diagnosis>tuberculose</diagnosis>")
            .replace("antibiothérapie", "<treatment>antibiothérapie</treatment>")
            .replace("pneumonie", "<diagnosis>pneumonie</diagnosis>")
            .replace("grave", "grave</diagnosis>")
        )
        return res

    with mock_llm_service(responder=responder):
        docs = edsnlp.data.from_iterable(md, converter="markup", preset="md")
        docs = docs.map(lambda x: x.text)
        docs = docs.map_pipeline(nlp)
        docs = docs.to_iterable(converter="markup", preset="md")
        docs = list(docs)
        assert docs == md


def test_async_worker_error():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.llm_markup_extractor(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            prompt=PROMPT,
            max_concurrent_requests=2,
        )
    )

    def responder(**kw):
        raise ValueError("Simulated error")

    with mock_llm_service(responder=responder), pytest.raises(RuntimeError):
        md = [
            "La patient souffre de tuberculose.",
            "On débute une antibiothérapie dès ajd.",
            "Il a une pneumonie du thorax.",
            "C'est très grave.",
        ]
        docs = edsnlp.data.from_iterable(md, converter="markup", preset="md")
        docs = docs.map(lambda x: x.text)
        docs = docs.map_pipeline(nlp)
        docs = docs.to_iterable(converter="markup", preset="md")
        docs = list(docs)
        assert docs == md


def test_context_getter_async():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe(
        eds.llm_markup_extractor(
            api_url="http://localhost:8080/v1",
            model="my-custom-model",
            prompt=PROMPT,
            max_concurrent_requests=2,
            context_getter="sents",
        )
    )

    md = [
        "La patient souffre de [tuberculose](diagnosis). On débute une "
        "[antibiothérapie](treatment) dès ajd.",
        "Il a une [pneumonie](diagnosis) du thorax. C'est très grave.",
    ]

    counter = 0

    def responder(messages, **kw):
        nonlocal counter
        counter += 1
        assert len(messages) == 2  # 1 system + 1 user
        res = (
            messages[-1]["content"]
            .replace("tuberculose", "<diagnosis>tuberculose</diagnosis>")
            .replace("antibiothérapie", "<treatment>antibiothérapie</treatment>")
            .replace("pneumonie", "<diagnosis>pneumonie</diagnosis>")
            .replace("grave", "grave</diagnosis>")
        )
        return res

    with mock_llm_service(responder=responder):
        docs = edsnlp.data.from_iterable(md, converter="markup", preset="md")
        docs = docs.map(lambda x: x.text)
        docs = docs.map_pipeline(nlp)
        docs = docs.to_iterable(converter="markup", preset="md")
        docs = list(docs)
        assert docs == md

    assert counter == 4
