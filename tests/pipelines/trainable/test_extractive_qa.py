import re

import pytest
from confit.utils.random import set_seed
from spacy.tokens import Span

import edsnlp
import edsnlp.pipes as eds

pytest.importorskip("torch.nn")


def normalize_decoded_trf_input(text: str) -> str:
    return re.sub(r"\s*'\s*", "'", text)


def test_ner():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.extractive_qa(
            embedding=eds.transformer(
                model="hf-internal-testing/tiny-random-bert",
                window=40,
                stride=20,
            ),
            # During training, where do we get the gold entities from ?
            target_span_getter=["ner-gold"],
            # Which prompts for each label ?
            questions={
                "PERSON": "Quels sont les personnages ?",
                "GIFT": "Quels sont les cadeaux ?",
            },
            questions_attribute="question",
            # During prediction, where do we set the predicted entities ?
            span_setter="ents",
        ),
    )

    doc = nlp(
        "L'aîné eut le Moulin, le second eut l'âne, et le plus jeune n'eut que le Chat."
    )
    doc._.question = {
        "FAVORITE": ["Qui a eu de l'argent ?"],
    }
    # doc[0:2], doc[4:5], doc[6:8], doc[9:11], doc[13:16], doc[20:21]
    doc.spans["ner-gold"] = [
        Span(doc, 0, 2, "PERSON"),  # L'aîné
        Span(doc, 4, 5, "GIFT"),  # Moulin
        Span(doc, 6, 8, "PERSON"),  # le second
        Span(doc, 9, 11, "GIFT"),  # l'âne
        Span(doc, 13, 16, "PERSON"),  # le plus jeune
        Span(doc, 20, 21, "GIFT"),  # Chat
    ]
    nlp.post_init([doc])

    ner = nlp.pipes.extractive_qa
    batch = ner.prepare_batch([doc], supervision=True)
    results = ner.module_forward(batch)

    list(ner.pipe([doc]))[0]

    assert results["loss"] is not None
    trf_inputs = [
        normalize_decoded_trf_input(seq.replace(" [PAD]", ""))
        for seq in ner.embedding.tokenizer.batch_decode(batch["embedding"]["input_ids"])
    ]
    assert trf_inputs == [
        "[CLS] quels sont les cadeaux? [SEP] l'aine eut le moulin, le second eut l'ane, et le p [SEP]",  # noqa: E501
        "[CLS] quels sont les cadeaux? [SEP] second eut l'ane, et le plus jeune n'eut que le ch [SEP]",  # noqa: E501
        "[CLS] quels sont les cadeaux? [SEP]lus jeune n'eut que le chat. [SEP]",
        "[CLS] quels sont les personnages? [SEP] l'aine eut le moulin, le second eut l'ane, et le p [SEP]",  # noqa: E501
        "[CLS] quels sont les personnages? [SEP] second eut l'ane, et le plus jeune n'eut que le ch [SEP]",  # noqa: E501
        "[CLS] quels sont les personnages? [SEP]lus jeune n'eut que le chat. [SEP]",
        "[CLS] qui a eu de l'argent? [SEP] l'aine eut le moulin, le second eut l'ane, et le p [SEP]",  # noqa: E501
        "[CLS] qui a eu de l'argent? [SEP] second eut l'ane, et le plus jeune n'eut que le ch [SEP]",  # noqa: E501
        "[CLS] qui a eu de l'argent? [SEP]lus jeune n'eut que le chat. [SEP]",
    ]
    assert batch["targets"].squeeze(2).tolist() == [
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        [2, 3, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 2, 1, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    assert nlp.config.to_yaml_str() == (
        "nlp:\n"
        "    lang: eds\n"
        "    pipeline:\n"
        "    - extractive_qa\n"
        "    tokenizer:\n"
        "        '@tokenizers': eds.tokenizer\n"
        "components:\n"
        "    extractive_qa:\n"
        "        '@factory': eds.extractive_qa\n"
        "        embedding:\n"
        "            '@factory': eds.transformer\n"
        "            model: hf-internal-testing/tiny-random-bert\n"
        "            window: 40\n"
        "            stride: 20\n"
        "        questions:\n"
        "            PERSON: Quels sont les personnages ?\n"
        "            GIFT: Quels sont les cadeaux ?\n"
        "        questions_attribute: question\n"
        "        target_span_getter:\n"
        "        - ner-gold\n"
        "        span_setter:\n"
        "            ents: true\n"
        "        infer_span_setter: false\n"
        "        mode: joint\n"
        "        window: 40\n"
        "        stride: 20\n"
    )


def test_confidence_score():
    set_seed(0)
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.extractive_qa(
            embedding=eds.transformer(
                model="hf-internal-testing/tiny-random-bert",
                window=40,
                stride=20,
            ),
            questions={"GIFT": "Quels sont les cadeaux ?"},
            target_span_getter=["ents"],
            span_setter="ents",
        ),
    )
    nlp.pipes.extractive_qa.compute_confidence_score = True

    nlp.train(False)
    doc = nlp("Le Chat dort.")

    assert doc.ents
    assert all(0 < ent._.ner_confidence_score < 1 for ent in doc.ents)
