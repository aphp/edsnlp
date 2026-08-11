import foldedtensor as ft
import pytest
from spacy.tokens import Span

import edsnlp
import edsnlp.pipes as eds
from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

pytestmark = pytest.mark.ml


class FakePromptEmbedding(WordEmbeddingComponent):
    """Small ragged embedding used to exercise the component without a model download"""

    def __init__(self, nlp=None, name="fake_prompt_embedding", output_size=8):
        super().__init__(nlp=nlp, name=name)
        self.output_size = output_size
        self.word_table = torch.nn.Embedding(256, output_size)
        self.prompt_table = torch.nn.Embedding(256, output_size)

    def add_special_tokens(self, tokens):
        self.role_markers = tokens

    def preprocess(self, doc, *, contexts, segments, **kwargs):
        return {
            "word_ids": [
                [1 + (token.norm % 255) for token in context] for context in contexts
            ],
            "prompt_ids": [
                [1 + sum(map(ord, prompt)) % 255 for prompt in prompts]
                for prompts in segments
            ],
        }

    def collate(self, batch):
        return {
            "word_ids": ft.as_folded_tensor(
                batch["word_ids"],
                data_dims=("word",),
                full_names=("sample", "context", "word"),
                dtype=torch.long,
            ),
            "segment_token_ids": ft.as_folded_tensor(
                [
                    [
                        [[prompt_id, prompt_id] for prompt_id in context_prompts]
                        for context_prompts in sample_prompts
                    ]
                    for sample_prompts in batch["prompt_ids"]
                ],
                data_dims=("segment_token",),
                full_names=(
                    "sample",
                    "context",
                    "segment",
                    "segment_token",
                ),
                dtype=torch.long,
            ),
        }

    def forward(self, batch):
        word_ids = batch["word_ids"]
        segment_token_ids = batch["segment_token_ids"]
        return {
            "embeddings": word_ids.with_data(self.word_table(word_ids.data)),
            "segment_embeddings": segment_token_ids.with_data(
                self.prompt_table(segment_token_ids.data)
            ),
        }


def test_overlap_bioul_union_and_bounds():
    from edsnlp.pipes.trainable.layers.crf import MultiLabelBIOULDecoder
    from edsnlp.pipes.trainable.multi_query_qa.multi_query_qa import (
        make_union_tags,
    )

    spans = torch.tensor([[0, 0, 6], [0, 2, 8]])
    tags, conflicts = make_union_tags(spans, num_contexts=1, max_words=8)

    assert tags.tolist() == [[2, 1, 2, 1, 1, 3, 1, 3]]
    assert conflicts == 0
    decoder = MultiLabelBIOULDecoder(1, allow_overlap=True)
    decoded = decoder.tags_to_spans(tags.unsqueeze(-1))[:, :3]
    assert decoded.tolist() == [
        [0, 0, 6],
        [0, 0, 8],
        [0, 2, 6],
        [0, 2, 8],
    ]
    _, conflicts = make_union_tags(
        torch.tensor([[0, 0, 3], [0, 2, 5]]),
        num_contexts=1,
        max_words=5,
    )
    assert conflicts == 1


def test_multi_query_qa_supervision_and_inference():
    nlp = edsnlp.blank("eds")
    component = eds.multi_query_qa(
        embedding=FakePromptEmbedding(),
        projection_size=8,
    )
    nlp.add_pipe(component)

    doc = nlp.make_doc("Biopsie hépatique puis IRM normale")
    biopsy = Span(doc, 0, 2)
    biopsy._.extraction_ids = ("biopsy", "procedure")
    biopsy._.facet_values = {
        "assertion": ("affirmed",),
        "context": ("planned",),
    }
    biopsy._.facet_known_values = {
        "assertion": ("negated", "affirmed"),
        "context": ("past", "planned"),
    }
    imaging = Span(doc, 3, 5)
    imaging._.extraction_ids = ("imaging",)
    imaging._.facet_values = {
        "assertion": None,
        "context": (),
    }
    imaging._.facet_known_values = {
        "assertion": (),
        "context": ("past", "planned"),
    }
    doc.spans["multi_query_qa_gold"] = [biopsy, imaging]
    doc._.queries = {
        "extractions": [
            {"id": "biopsy", "prompt": "biopsie"},
            {"id": "procedure", "prompt": "procédure"},
            {"id": "imaging", "prompt": "imagerie"},
        ],
        "classifications": [
            {
                "facet": "assertion",
                "cardinality": "single",
                "options": [
                    {"value_id": "negated", "prompt": "assertion, nié"},
                    {"value_id": "affirmed", "prompt": "assertion, affirmé"},
                ],
            },
            {
                "facet": "context",
                "cardinality": "multi",
                "options": [
                    {"value_id": "past", "prompt": "contexte, passé"},
                    {"value_id": "planned", "prompt": "contexte, prévu"},
                ],
            },
        ],
    }

    batch = component.prepare_batch([doc], supervision=True)
    assert batch["extraction_prompt_indices"].tolist() == [0, 1, 2]
    assert batch["classification_option_prompt_indices"].tolist() == [3, 4, 5, 6]
    assert batch["gold_tags"].tolist() == [[2, 3, 0, 2, 3]]
    assert batch["gold_extraction_pairs"].tolist() == [[0, 0], [0, 1], [1, 2]]
    assert batch["gold_single_targets"].tolist() == [1, -100, -100, -100]
    assert batch["gold_multi_targets"].tolist() == [
        -100,
        -100,
        0,
        1,
        -100,
        -100,
        0,
        0,
    ]

    output = component.module_forward(batch)
    assert output["loss"].isfinite()
    assert output["candidate_spans"].shape[1] == 3
    assert {(0, 0, 2), (0, 3, 5)} <= set(map(tuple, output["candidate_spans"].tolist()))
    output["loss"].backward()
    assert component.head.span_projection.weight.grad.isfinite().all()

    no_answer = nlp.make_doc("Aucune anomalie retrouvée")
    no_answer.spans["multi_query_qa_gold"] = []
    no_answer._.queries = doc._.queries
    mixed_batch = component.prepare_batch([doc, no_answer], supervision=True)
    mixed_output = component.module_forward(mixed_batch)
    assert mixed_batch["extraction_context_indices"].tolist() == [0, 0, 0, 1, 1, 1]
    assert mixed_output["loss"].isfinite()

    component.eval()
    with torch.no_grad():
        component.head.candidate_linear.weight.zero_()
        component.head.candidate_linear.bias.zero_()
        component.head.candidate_linear.bias[4] = 10
        component.head.span_projection.weight.zero_()
        component.head.span_projection.bias.zero_()
        component.head.query_projection.weight.zero_()
        component.head.query_projection.bias.zero_()
        component.head.score_bias.fill_(10)

    inferred = list(component.pipe([doc.copy()]))[0]
    predicted = inferred.spans["multi_query_qa"]
    assert len(predicted) == len(doc) * (len(doc) + 1) // 2 * 3
    assert {span._.extraction_id for span in predicted} == {
        "biopsy",
        "procedure",
        "imaging",
    }
    biopsy_prediction = next(
        span
        for span in predicted
        if span._.extraction_id == "biopsy" and (span.start, span.end) == (0, 2)
    )
    assert biopsy_prediction.text == "Biopsie hépatique"
    assert (biopsy_prediction.start_char, biopsy_prediction.end_char) == (0, 17)
    assert all(
        span._.facet_values
        == {
            "assertion": ("negated",),
            "context": ("past", "planned"),
        }
        for span in predicted
    )

    empty = nlp.make_doc("Aucune anomalie")
    empty.spans["multi_query_qa_gold"] = []
    empty._.queries = {
        "extractions": [{"id": "finding", "prompt": "anomalie"}],
        "classifications": [],
    }
    with torch.no_grad():
        component.head.candidate_linear.bias.zero_()
        component.head.candidate_linear.bias[0] = 10
    empty_batch = component.prepare_batch([empty], supervision=True)
    empty_output = component.module_forward(empty_batch)
    assert empty_output["candidate_spans"].shape == (0, 3)
    assert empty_output["loss"].isfinite()
    assert not list(component.pipe([empty.copy()]))[0].spans["multi_query_qa"]


def test_multi_query_qa_prompt_batch_sizes():
    realistic_prompts = [
        "biopsie",
        "imagerie médicale",
        "intervention chirurgicale",
        "infection bactérienne",
        "tumeur maligne",
        "douleur thoracique",
        "traitement antibiotique",
        "insuffisance cardiaque",
        "fracture osseuse",
        "examen biologique",
        "dispositif médical",
        "greffe d'organe",
        "hémorragie digestive",
        "maladie rénale",
        "réaction allergique",
        "radiothérapie",
    ]
    nlp = edsnlp.blank("eds")
    component = eds.multi_query_qa(
        embedding=FakePromptEmbedding(),
        projection_size=8,
    )
    nlp.add_pipe(component)

    for num_extractions in (1, 8, 16, 32, 128):
        doc = nlp.make_doc("Biopsie hépatique réalisée")
        gold = Span(doc, 0, 2)
        gold._.extraction_ids = ("q0",)
        gold._.facet_values = {}
        doc.spans["multi_query_qa_gold"] = [gold]
        prompts = realistic_prompts + [
            f"concept clinique {idx}"
            for idx in range(len(realistic_prompts), num_extractions)
        ]
        doc._.queries = {
            "extractions": [
                {"id": f"q{idx}", "prompt": prompt}
                for idx, prompt in enumerate(prompts[:num_extractions])
            ],
            "classifications": [],
        }

        batch = component.prepare_batch([doc], supervision=True)
        output = component.module_forward(batch)
        assert (
            len(
                batch["embedding"]["segment_token_ids"]
                .refold("segment", "segment_token")
                .as_tensor()
            )
            == num_extractions
        )
        assert len(output["extraction_pair_queries"]) == (
            len(output["candidate_spans"]) * num_extractions
        )
        assert output["loss"].isfinite()


def test_multi_query_qa_head_tiny_overfit():
    from edsnlp.pipes.trainable.multi_query_qa.multi_query_qa import MultiQueryQAHead

    torch.manual_seed(0)
    head = MultiQueryQAHead(input_size=6, projection_size=8)
    words = torch.randn(5, 6)
    dense_words = words.unsqueeze(0)
    spans = torch.tensor([[0, 0, 2], [0, 3, 5]])
    word_offsets = torch.tensor([0, 5])
    prompt_embeddings = torch.randn(2, 3, 6)
    prompt_mask = torch.ones((2, 3), dtype=torch.bool)
    pair_spans = torch.tensor([0, 0, 1, 1])
    pair_prompts = torch.tensor([0, 1, 0, 1])
    pair_targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
    tag_targets = torch.tensor([[2, 3, 0, 2, 3]])
    mask = torch.ones((1, 5), dtype=torch.bool)

    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    initial_loss = None
    for _ in range(100):
        optimizer.zero_grad()
        emissions = head.candidate_linear(dense_words)
        candidate_loss = head.crf(
            emissions,
            mask,
            F.one_hot(tag_targets, 5).bool(),
        ).mean()
        scores = head.score_pairs(
            head.encode_spans(words, spans, word_offsets),
            head.encode_queries(
                prompt_embeddings,
                prompt_mask,
            ),
            pair_spans,
            pair_prompts,
        )
        loss = candidate_loss + F.binary_cross_entropy_with_logits(
            scores,
            pair_targets,
        )
        initial_loss = loss.item() if initial_loss is None else initial_loss
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss * 0.05
    assert torch.equal(
        head.crf.decode(head.candidate_linear(dense_words), mask), tag_targets
    )
    assert torch.equal(scores.sigmoid() >= 0.5, pair_targets.bool())


def test_multi_query_qa_head_uses_role_marker():
    from edsnlp.pipes.trainable.multi_query_qa.multi_query_qa import MultiQueryQAHead

    head = MultiQueryQAHead(input_size=2, projection_size=2)
    with torch.no_grad():
        head.query_projection.weight.copy_(torch.eye(2))
        head.query_projection.bias.zero_()

    queries = head.encode_queries(
        torch.tensor([[[1.0, 3.0], [3.0, 5.0]], [[7.0, 9.0], [0.0, 0.0]]]),
        torch.tensor([[True, True], [True, False]]),
    )

    assert queries.tolist() == [[1.0, 3.0], [7.0, 9.0]]
