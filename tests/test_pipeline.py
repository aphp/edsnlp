from io import BytesIO
from itertools import chain

import pytest
from confit import Config
from confit.errors import ConfitValidationError
from confit.registry import validate_arguments
from spacy.tokens import Doc

import edsnlp
import edsnlp.accelerators.multiprocessing
from edsnlp import Pipeline, registry
from edsnlp.pipelines.factories import normalizer, sentences


class CustomClass:
    pass

    def __call__(self, doc: Doc) -> Doc:
        return doc


def make_pipeline():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.sentences", name="sentences")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
        ),
    )
    nlp.add_pipe(
        "eds.ner_crf",
        name="ner",
        config=dict(
            embedding=nlp.get_pipe("transformer"),
            mode="independent",
            target_span_getter=["ents", "ner-preds"],
            span_setter="ents",
        ),
    )
    ner = nlp.get_pipe("ner")
    ner.update_labels(["PERSON", "GIFT"])
    return nlp


@pytest.fixture()
def pipeline():
    return make_pipeline()


@pytest.fixture(scope="session")
def frozen_pipeline():
    return make_pipeline()


def test_add_pipe_factory():
    model = edsnlp.blank("eds")
    model.add_pipe("eds.normalizer", name="normalizer")
    assert "normalizer" in model.pipe_names
    assert model.has_pipe("normalizer")

    model.add_pipe("eds.sentences", name="sentences")
    assert "sentences" in model.pipe_names
    assert model.has_pipe("sentences")

    with pytest.raises(ValueError):
        model.get_pipe("missing-pipe")


def test_add_pipe_component():
    model = edsnlp.blank("eds")
    model.add_pipe(normalizer(nlp=model, name="normalizer"))
    assert "normalizer" in model.pipe_names
    assert model.has_pipe("normalizer")

    model.add_pipe(sentences(nlp=model, name="sentences"))
    assert "sentences" in model.pipe_names
    assert model.has_pipe("sentences")

    with pytest.raises(ValueError):
        model.add_pipe(
            sentences(nlp=model, name="sentences"),
            config={"punct_chars": ".?!"},
        )

    with pytest.raises(ValueError):
        model.add_pipe(CustomClass())


def test_sequence(frozen_pipeline: Pipeline):
    assert len(frozen_pipeline.pipeline) == 3
    assert list(frozen_pipeline.pipeline) == [
        ("sentences", frozen_pipeline.get_pipe("sentences")),
        ("transformer", frozen_pipeline.get_pipe("transformer")),
        ("ner", frozen_pipeline.get_pipe("ner")),
    ]
    assert list(frozen_pipeline.torch_components()) == [
        ("transformer", frozen_pipeline.get_pipe("transformer")),
        ("ner", frozen_pipeline.get_pipe("ner")),
    ]


def test_disk_serialization(tmp_path, pipeline):
    nlp = pipeline

    ner = nlp.get_pipe("ner")
    ner.update_labels(["PERSON", "GIFT"])
    nlp.to_disk(tmp_path / "model")

    assert (tmp_path / "model" / "config.cfg").exists()
    assert (tmp_path / "model" / "tensors" / "ner+transformer.safetensors").exists()

    assert (tmp_path / "model" / "config.cfg").read_text() == config_str.replace(
        "components = ${components}\n", ""
    )

    nlp = edsnlp.blank(
        "eds", config=Config.from_disk(tmp_path / "model" / "config.cfg")
    )
    nlp.from_disk(tmp_path / "model")
    assert nlp.get_pipe("ner").labels == ["PERSON", "GIFT"]


config_str = """\
[nlp]
lang = "eds"
pipeline = ["sentences", "transformer", "ner"]
components = ${components}
disabled = []

[nlp.tokenizer]
@tokenizers = "eds.tokenizer"

[components]

[components.sentences]
@factory = "eds.sentences"

[components.transformer]
@factory = "eds.transformer"
model = "prajjwal1/bert-tiny"
window = 128
stride = 96

[components.ner]
@factory = "eds.ner_crf"
embedding = ${components.transformer}
mode = "independent"
target_span_getter = ["ents", "ner-preds"]
labels = ["PERSON", "GIFT"]
infer_span_setter = false
window = 20
stride = 18

[components.ner.span_setter]
ents = true

"""


def test_validate_config():
    @validate_arguments
    def function(model: Pipeline):
        print(model.pipe_names)
        assert len(model.pipe_names) == 3

    function(Config.from_str(config_str).resolve(registry=registry)["nlp"])


def test_torch_module(frozen_pipeline: Pipeline):
    with frozen_pipeline.train(True):
        for name, component in frozen_pipeline.torch_components():
            assert component.training is True

    with frozen_pipeline.train(False):
        for name, component in frozen_pipeline.torch_components():
            assert component.training is False

    frozen_pipeline.to("cpu")


def test_cache(frozen_pipeline: Pipeline):
    text = "Ceci est un exemple"
    frozen_pipeline(text)

    with frozen_pipeline.cache():
        frozen_pipeline(text)
        trf_forward_cache_entries = [
            key
            for key in frozen_pipeline._cache
            if isinstance(key, tuple) and key[:2] == ("transformer", "forward")
        ]
        assert len(trf_forward_cache_entries) == 1

    assert frozen_pipeline._cache is None


def test_select_pipes(pipeline: Pipeline):
    text = "Ceci est un exemple"
    with pipeline.select_pipes(enable=["transformer", "ner"]):
        assert not pipeline(text).has_annotation("SENT_START")


def test_different_names():
    nlp = edsnlp.blank("eds")

    extractor = sentences(nlp=nlp, name="custom_name")

    with pytest.raises(ValueError) as exc_info:
        nlp.add_pipe(extractor, name="sentences")

    assert "The provided name does not match the name of the component." in str(
        exc_info.value
    )


fail_config = """
[nlp]
lang = "eds"
pipeline = ["transformer", "ner"]
disabled = []

[nlp.tokenizer]
@tokenizers = "eds.tokenizer"

[components]

[components.transformer]
@factory = "eds.transformer"
model = "prajjwal1/bert-tiny"
window = 128
stride = 96

[components.ner]
@factory = "eds.ner_crf"
embedding = ${components.transformer}
mode = "error-mode"
span_setter = "ents"
"""


def test_config_validation_error():
    with pytest.raises(ConfitValidationError) as e:
        Pipeline.from_config(Config.from_str(fail_config))

    assert str(e.value) == (
        "1 validation error for edsnlp.core.pipeline.Pipeline()\n"
        "-> components.ner.mode\n"
        "   unexpected value; permitted: 'independent', 'joint', 'marginal', got "
        "'error-mode' (str)"
    )


def test_add_pipe_validation_error():
    model = edsnlp.blank("eds")
    with pytest.raises(ConfitValidationError) as e:
        model.add_pipe("eds.covid", name="extractor", config={"foo": "bar"})

    assert str(e.value) == (
        "1 validation error for "
        "edsnlp.pipelines.ner.covid.factory.create_component()\n"
        "-> extractor.foo\n"
        "   unexpected keyword argument"
    )


def test_multiprocessing_accelerator(frozen_pipeline):
    texts = ["Ceci est un exemple", "Ceci est un autre exemple"]
    edsnlp.accelerators.multiprocessing.MAX_NUM_PROCESSES = 2
    docs = list(
        frozen_pipeline.pipe(
            texts * 20,
            accelerator="multiprocessing",
            batch_size=2,
        )
    )
    assert len(docs) == 40


def error_pipe(doc: Doc):
    if doc._.note_id == "text-3":
        raise ValueError("error")
    return doc


def test_multiprocessing_gpu_stub(frozen_pipeline):
    text1 = "Ceci est un exemple"
    text2 = "Ceci est un autre exemple"
    edsnlp.accelerators.multiprocessing.MAX_NUM_PROCESSES = 2
    accelerator = edsnlp.accelerators.multiprocessing.MultiprocessingAccelerator(
        batch_size=2,
        num_gpu_workers=1,
        num_cpu_workers=1,
        gpu_worker_devices=["cpu"],
    )
    list(
        frozen_pipeline.pipe(
            chain.from_iterable(
                [
                    {"content": text1},
                    {"content": text2},
                ]
                for i in range(5)
            ),
            accelerator=accelerator,
            to_doc="content",
            from_doc={"ents": "ents"},
        )
    )


def test_multiprocessing_rb_error(pipeline):
    text1 = "Ceci est un exemple"
    text2 = "Ceci est un autre exemple"
    edsnlp.accelerators.multiprocessing.MAX_NUM_PROCESSES = 2
    pipeline.add_pipe(error_pipe, name="error", after="sentences")
    with pytest.raises(ValueError):
        list(
            pipeline.pipe(
                chain.from_iterable(
                    [
                        {"content": text1, "id": f"text-{i}"},
                        {"content": text2, "id": f"other-text-{i}"},
                    ]
                    for i in range(5)
                ),
                accelerator="multiprocessing",
                batch_size=2,
                to_doc={"text_field": "content", "id_field": "id"},
            )
        )


try:
    import torch

    from edsnlp.core.torch_component import TorchComponent

    class DeepLearningError(TorchComponent):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def preprocess(self, doc):
            return {"num_words": len(doc), "doc_id": doc._.note_id}

        def collate(self, batch):
            return {
                "num_words": torch.tensor(batch["num_words"]),
                "doc_id": batch["doc_id"],
            }

        def forward(self, batch):
            if "text-1" in batch["doc_id"]:
                raise RuntimeError("Deep learning error")
            return {}

except ImportError:
    pass


def test_multiprocessing_ml_error(pipeline):
    text1 = "Ceci est un exemple"
    text2 = "Ceci est un autre exemple"
    edsnlp.accelerators.multiprocessing.MAX_NUM_PROCESSES = 2
    pipeline.add_pipe(
        DeepLearningError(
            pipeline=pipeline,
            name="error",
        ),
        after="sentences",
    )
    accelerator = edsnlp.accelerators.multiprocessing.MultiprocessingAccelerator(
        batch_size=2,
        num_gpu_workers=1,
        num_cpu_workers=1,
        gpu_worker_devices=["cpu"],
    )
    with pytest.raises(RuntimeError) as e:
        list(
            pipeline.pipe(
                chain.from_iterable(
                    [
                        {"content": text1, "id": f"text-{i}"},
                        {"content": text2, "id": f"other-text-{i}"},
                    ]
                    for i in range(5)
                ),
                accelerator=accelerator,
                to_doc={"text_field": "content", "id_field": "id"},
            )
        )
    assert "Deep learning error" in str(e.value)


def test_spacy_component():
    nlp = edsnlp.blank("fr")
    nlp.add_pipe("sentencizer")


def test_rule_based_pipeline():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.covid")

    assert nlp.pipe_names == ["eds.normalizer", "eds.covid"]
    assert nlp.get_pipe("eds.normalizer") == nlp.pipeline[0][1]
    assert nlp.has_pipe("eds.covid")

    with pytest.raises(ValueError) as exc_info:
        nlp.get_pipe("unknown")

    assert str(exc_info.value) == "Pipe 'unknown' not found in pipeline."

    doc = nlp.make_doc("Mon patient a le covid")

    new_doc = nlp(doc)

    assert len(doc.ents) == 1
    assert new_doc is doc

    assert nlp.get_pipe_meta("eds.covid").assigns == ["doc.ents", "doc.spans"]


def test_torch_save(pipeline):
    import torch

    pipeline.get_pipe("ner").update_labels(["LOC", "PER"])
    buffer = BytesIO()
    torch.save(pipeline, buffer)
    buffer.seek(0)
    nlp = torch.load(buffer)
    assert nlp.get_pipe("ner").labels == ["LOC", "PER"]
    assert len(list(nlp("Une phrase. Deux phrases.").sents)) == 2
