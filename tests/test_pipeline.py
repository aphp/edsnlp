from io import BytesIO

import pytest
from confit import Config
from confit.errors import ConfitValidationError
from confit.registry import validate_arguments
from spacy.tokens import Doc

import edsnlp
from edsnlp import Pipeline, registry
from edsnlp.pipes.factories import normalizer, sentences


class CustomClass:
    pass

    def __call__(self, doc: Doc) -> Doc:
        return doc


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
    model.add_pipe(normalizer(nlp=model), name="normalizer")
    assert "normalizer" in model.pipe_names
    assert model.has_pipe("normalizer")

    model.add_pipe(sentences(nlp=model), name="sentences")
    assert "sentences" in model.pipe_names
    assert model.has_pipe("sentences")

    with pytest.raises(ValueError):
        model.add_pipe(
            sentences(nlp=model, name="sentences"),
            config={"punct_chars": ".?!"},
        )

    with pytest.raises(ValueError):
        model.add_pipe(CustomClass())


def test_sequence(frozen_ml_nlp: Pipeline):
    assert len(frozen_ml_nlp.pipeline) == 3
    assert list(frozen_ml_nlp.pipeline) == [
        ("sentences", frozen_ml_nlp.get_pipe("sentences")),
        ("transformer", frozen_ml_nlp.get_pipe("transformer")),
        ("ner", frozen_ml_nlp.get_pipe("ner")),
    ]
    assert list(frozen_ml_nlp.torch_components()) == [
        ("transformer", frozen_ml_nlp.get_pipe("transformer")),
        ("ner", frozen_ml_nlp.get_pipe("ner")),
    ]


def test_disk_serialization(tmp_path, ml_nlp):
    nlp = ml_nlp

    assert nlp.get_pipe("transformer").stride == 96
    ner = nlp.get_pipe("ner")
    ner.update_labels(["PERSON", "GIFT"])
    nlp.to_disk(tmp_path / "model")

    print("tmp_path", tmp_path, list((tmp_path / "model/transformer").iterdir()))

    assert (tmp_path / "model" / "config.cfg").exists()
    assert (tmp_path / "model" / "ner" / "parameters.safetensors").exists()
    assert (tmp_path / "model" / "transformer" / "parameters.safetensors").exists()
    # fmt: off
    assert (
          (tmp_path / "model" / "transformer" / "pytorch_model.bin").exists() or
          (tmp_path / "model" / "transformer" / "model.safetensors").exists()
    )
    # fmt: on

    assert (tmp_path / "model" / "config.cfg").read_text() == (
        config_str.replace("components = ${components}\n", "").replace(
            "prajjwal1/bert-tiny", "./transformer"
        )
    )

    nlp = edsnlp.load(
        tmp_path / "model",
        overrides={"components": {"transformer": {"stride": 64}}},
    )
    assert nlp.get_pipe("ner").labels == ["PERSON", "GIFT"]
    assert nlp.get_pipe("transformer").stride == 64


config_str = """\
[nlp]
lang = "eds"
pipeline = ["sentences", "transformer", "ner"]
components = ${components}

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
window = 40
stride = 20

[components.ner.span_setter]
ents = true

"""


def test_validate_config():
    @validate_arguments
    def function(model: Pipeline):
        print(model.pipe_names)
        assert len(model.pipe_names) == 3

    function(Config.from_str(config_str).resolve(registry=registry)["nlp"])


def test_torch_module(frozen_ml_nlp: Pipeline):
    with frozen_ml_nlp.train(True):
        for name, component in frozen_ml_nlp.torch_components():
            assert component.training is True

    with frozen_ml_nlp.train(False):
        for name, component in frozen_ml_nlp.torch_components():
            assert component.training is False

    frozen_ml_nlp.to("cpu")


def test_cache(frozen_ml_nlp: Pipeline):
    from edsnlp.core.torch_component import _caches

    text = "Ceci est un exemple"
    frozen_ml_nlp(text)

    doc = frozen_ml_nlp.make_doc(text)
    with frozen_ml_nlp.cache():
        for name, pipe in frozen_ml_nlp.pipeline:
            # This is a hack to get around the ambiguity
            # between the __call__ method of Pytorch modules
            # and the __call__ methods of spacy components
            if hasattr(pipe, "batch_process"):
                doc = next(iter(pipe.batch_process([doc])))
            else:
                doc = pipe(doc)
        trf_forward_cache_entries = [
            key
            for key in _caches["default"]
            if isinstance(key, tuple) and key[0] == "forward"
        ]
        assert len(trf_forward_cache_entries) == 2

    assert len(_caches) == 0


def test_select_pipes(frozen_ml_nlp: Pipeline):
    text = "Ceci est un exemple"
    with frozen_ml_nlp.select_pipes(enable=["transformer", "ner"]):
        assert len(frozen_ml_nlp.disabled) == 1
        assert not frozen_ml_nlp(text).has_annotation("SENT_START")
    assert len(frozen_ml_nlp.disabled) == 0


@pytest.mark.skip(reason="Deprecated behavior")
def test_different_names():
    nlp = edsnlp.blank("eds")

    extractor = sentences(nlp=nlp, name="custom_name")

    with pytest.raises(ValueError) as exc_info:
        nlp.add_pipe(extractor, name="sentences")

    assert (
        "The provided name 'sentences' does not "
        "match the name of the component 'custom_name'."
    ) in str(exc_info.value)


def test_load_config(run_in_test_dir):
    nlp = edsnlp.load("training/qlf_config.cfg")
    assert nlp.pipe_names == [
        "normalizer",
        "sentencizer",
        "embedding",
        "covid",
        "qualifier",
    ]


fail_config = """
[nlp]
lang = "eds"
pipeline = ["transformer", "ner"]

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

    assert "1 validation error for edsnlp.core.pipeline.Pipeline()" in str(e.value)
    assert "got 'error-mode'" in str(e.value)


def test_add_pipe_validation_error():
    model = edsnlp.blank("eds")
    with pytest.raises(ConfitValidationError) as e:
        model.add_pipe("eds.covid", name="extractor", config={"foo": "bar"})

    assert str(e.value) == (
        "1 validation error for "
        "edsnlp.pipes.ner.covid.factory.create_component()\n"
        "-> extractor.foo\n"
        "   unexpected keyword argument"
    )


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


def test_torch_save(ml_nlp):
    import torch

    ml_nlp.get_pipe("ner").update_labels(["LOC", "PER"])
    buffer = BytesIO()
    torch.save(ml_nlp, buffer)
    buffer.seek(0)
    nlp = torch.load(buffer)
    assert nlp.get_pipe("ner").labels == ["LOC", "PER"]
    assert len(list(nlp("Une phrase. Deux phrases.").sents)) == 2


def test_parameters(frozen_ml_nlp):
    assert len(list(frozen_ml_nlp.parameters())) == 40


def test_missing_factory(nlp):
    with pytest.raises(ValueError) as exc_info:
        nlp.add_pipe("__test_missing_pipe__")

    assert "__test_missing_pipe__" in str(exc_info.value)
