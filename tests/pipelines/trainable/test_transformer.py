import pytest
from pytest import fixture
from spacy.tokens import Span

import edsnlp
from edsnlp.utils.collections import batch_compress_dict, decompress_dict

pytestmark = pytest.mark.ml

if not Span.has_extension("label"):
    Span.set_extension("label", default=None)

if not Span.has_extension("event_type"):
    Span.set_extension("event_type", default=None)

if not Span.has_extension("test_negated"):
    Span.set_extension("test_negated", default=False)

torch = pytest.importorskip("torch")


@fixture
def gold():
    blank_nlp = edsnlp.blank("eds")
    doc1 = blank_nlp.make_doc("Arret du ttt si folfox inefficace. Une autre phrase.")

    doc1.spans["sc"] = [
        Span(doc1, 4, 5, "drug"),  # "folfox"
        Span(doc1, 0, 1, "event"),  # "Arret"
        Span(doc1, 3, 4, "criteria"),  # "si"
    ]
    doc1.spans["sc"][0]._.test_negated = False
    doc1.spans["sc"][1]._.test_negated = True
    doc1.spans["sc"][2]._.test_negated = False
    doc1.spans["sc"][1]._.event_type = "stop"
    doc1.spans["to_embed"] = [doc1[0:5], doc1[7:11]]

    doc1.spans["sent"] = [Span(doc1, 0, 6, "sent")]

    return [doc1]


def test_span_getter(gold):
    from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer
    from edsnlp.pipes.trainable.span_classifier.span_classifier import (
        TrainableSpanClassifier,
    )

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="hf-internal-testing/tiny-random-bert",
            window=128,
            stride=96,
            quantization=None,
        ),
    )
    nlp.add_pipe(
        "eds.span_qualifier",
        name="qualifier",
        config={
            "embedding": {
                "@factory": "eds.span_pooler",
                "embedding": nlp.get_pipe("transformer"),
            },
            "span_getter": ["ents", "sc"],
            "context_getter": ["to_embed"],
            "qualifiers": ["_.test_negated", "_.event_type"],
        },
    )
    trf: Transformer = nlp.get_pipe("transformer")
    qlf: TrainableSpanClassifier = nlp.get_pipe("qualifier")
    qlf.post_init(gold, set())
    batch = qlf.prepare_batch([doc.copy() for doc in gold], supervision=True)
    input_ids = batch["embedding"]["embedding"]["input_ids"]
    mask = input_ids.mask
    tok = trf.tokenizer
    assert len(input_ids) == 2
    assert tok.decode(input_ids[0][mask[0]]) == "[CLS] arret du ttt si folfox [SEP]"
    assert tok.decode(input_ids[1][mask[1]]) == "[CLS] une autre phrase. [SEP]"

    # Transformer alone with prompts (usually passed by the caller component)
    prep = trf.preprocess(
        gold[0],
        contexts=gold[0].spans["to_embed"],
        prompts=["drug", "drug"],
    )
    batch = decompress_dict(list(batch_compress_dict([prep])))
    batch = trf.collate(batch)
    batch = trf.batch_to_device(batch, device=trf.device)
    res = trf(batch)
    assert res["embeddings"].shape == (2, 5, 32)
    assert "prompt_embeddings" not in res


def test_preprocess_suppresses_transformers_sequence_length_warning(caplog):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="hf-internal-testing/tiny-random-bert",
            window=128,
            stride=96,
            quantization=None,
        ),
    )
    trf = nlp.pipes.transformer
    trf.tokenizer.model_max_length = 5

    doc = nlp.make_doc("Michael Scott is a man of few words.")
    with caplog.at_level("WARNING", logger="transformers.tokenization_utils_base"):
        trf.preprocess(doc, prompts=["clinical entity"])

    assert not any(
        "Token indices sequence length is longer than the specified maximum"
        in record.message
        for record in caplog.records
    )


def test_segments_share_document_encoding(gold):
    from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer

    nlp = edsnlp.blank("eds")
    trf = Transformer(
        nlp,
        model="hf-internal-testing/tiny-random-bert",
        window=128,
        stride=96,
        max_tokens_per_device=1_000_000,
    )
    role_markers = ["[EXTRACT]", "[CLASSIFY]"]
    trf.add_special_tokens(role_markers)
    assert all(
        len(trf.tokenizer(marker, add_special_tokens=False).input_ids) == 1
        for marker in role_markers
    )
    segments = [
        "[EXTRACT] folfox | oxaliplatin",
        "[CLASSIFY] assertion: affirmed",
        "[CLASSIFY] assertion: negated",
    ]
    prep = trf.preprocess(
        gold[0],
        contexts=[gold[0][:]],
        segments=[segments],
    )
    batch = decompress_dict(list(batch_compress_dict([prep])))
    batch = trf.batch_to_device(trf.collate(batch), device=trf.device)

    decoded = [
        trf.tokenizer.decode(row[mask])
        for row, mask in zip(batch["input_ids"], batch["input_ids"].mask)
    ]
    assert decoded == [
        "[CLS] [EXTRACT] folfox | oxaliplatin [SEP] [CLASSIFY] assertion : "
        "affirmed [SEP] [CLASSIFY] assertion : negated [SEP] arret du "
        "ttt si folfox inefficace. une autre phrase. [SEP]"
    ]

    encoder_shapes = []
    hook = trf.transformer.base_model.register_forward_hook(
        lambda _, args, output: encoder_shapes.append(
            output.last_hidden_state.shape[:2]
        )
    )

    output = trf(batch)
    hook.remove()
    assert encoder_shapes == [batch["input_ids"].shape]
    assert output["embeddings"].shape == (1, len(gold[0]), 32)
    segment_embeddings = output["segment_embeddings"]
    assert segment_embeddings.full_names == (
        "sample",
        "context",
        "segment",
        "segment_token",
    )
    assert segment_embeddings.refold("segment", "segment_token").shape == (
        3,
        max(map(len, prep["segments"][0])),
        32,
    )
    assert batch["segment_indices"].full_names == (
        "sample",
        "context",
        "segment",
        "segment_token",
    )
    assert output["prompt_embeddings"].full_names == (
        "sample",
        "context",
        "prompt_segment",
        "prompt_token",
    )
    assert output["prompt_embeddings"].requires_grad

    with pytest.warns(DeprecationWarning, match="prompt_segments"):
        legacy = trf.preprocess(
            gold[0], contexts=[gold[0][:]], prompt_segments=[segments]
        )
    assert legacy["segments"] == prep["segments"]
    with pytest.raises(ValueError, match="cannot be combined"):
        trf.preprocess(
            gold[0],
            contexts=[gold[0][:]],
            segments=[segments],
            prompt_segments=[segments],
        )


def test_segments_repeat_over_document_windows(gold):
    from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer

    nlp = edsnlp.blank("eds")
    trf = Transformer(
        nlp,
        model="hf-internal-testing/tiny-random-bert",
        window=8,
        stride=4,
        max_tokens_per_device=1_000_000,
    )
    segments = ["procedure", "biopsy"]
    prep = trf.preprocess(gold[0], contexts=[gold[0][:]], segments=[segments])
    prefix_size = 1 + sum(len(segment) + 1 for segment in prep["segments"][0])
    trf.transformer.config.max_position_embeddings = prefix_size + 5
    batch = decompress_dict(list(batch_compress_dict([prep])))
    batch = trf.batch_to_device(trf.collate(batch), device=trf.device)

    prefix = "[CLS] procedure [SEP] biopsy [SEP]"
    decoded = [
        trf.tokenizer.decode(row[mask])
        for row, mask in zip(batch["input_ids"], batch["input_ids"].mask)
    ]
    assert len(decoded) > 1
    assert all(sequence.startswith(prefix) for sequence in decoded)
    assert batch["input_ids"].shape[1] <= prefix_size + 5
    hidden_states = []
    hook = trf.transformer.base_model.register_forward_hook(
        lambda _, args, output: hidden_states.append(output.last_hidden_state)
    )
    output = trf(batch)
    hook.remove()
    assert output["embeddings"].shape == (1, len(gold[0]), 32)
    assert output["embeddings"].data.isfinite().all()
    segment_embeddings = output["segment_embeddings"]
    segments = segment_embeddings.refold("segment", "segment_token")
    assert segments.shape == (
        2,
        max(map(len, prep["segments"][0])),
        32,
    )
    assert segment_embeddings.full_names == (
        "sample",
        "context",
        "segment",
        "segment_token",
    )
    raw_segment_embeddings = hidden_states[0].flatten(0, 1)[
        batch["segment_token_indices"]
    ]
    first_token_occurrences = raw_segment_embeddings[
        batch["segment_token_groups"] == 0
    ]
    assert len(first_token_occurrences) == len(decoded)
    assert not torch.allclose(first_token_occurrences[0], first_token_occurrences[-1])
    assert torch.allclose(segment_embeddings.data[0], first_token_occurrences.mean(0))

    trf.transformer.config.max_position_embeddings = prefix_size + 1
    prep = trf.preprocess(
        gold[0], contexts=[gold[0][:]], segments=[["procedure", "biopsy"]]
    )
    batch = decompress_dict(list(batch_compress_dict([prep])))
    with pytest.raises(ValueError, match="no Transformer position"):
        trf.collate(batch)


def test_segment_batch_sizes(gold):
    from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer

    trf = Transformer(
        edsnlp.blank("eds"),
        model="hf-internal-testing/tiny-random-bert",
        window=128,
        stride=96,
        max_tokens_per_device=1_000_000,
    )
    realistic = [
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
    sizes = (1, 8, 16, 32, 128)
    preps = [
        trf.preprocess(
            gold[0],
            contexts=[gold[0][:]],
            segments=[
                [
                    *(realistic[: min(size, len(realistic))]),
                    *("x" for _ in range(len(realistic), size)),
                ]
            ],
        )
        for size in sizes
    ]
    batch = decompress_dict(list(batch_compress_dict(preps)))
    batch = trf.batch_to_device(trf.collate(batch), device=trf.device)

    output = trf(batch)

    assert len(batch["input_ids"]) >= len(sizes)
    segments = output["segment_embeddings"].refold("segment", "segment_token")
    assert len(segments) == sum(sizes)
