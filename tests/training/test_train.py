# ruff:noqa:E402

import pytest

try:
    import torch.nn
except ImportError:
    torch = None

if torch is None:
    pytest.skip("torch not installed", allow_module_level=True)
pytest.importorskip("rich")

import shutil
from typing import (
    Optional,
    Sequence,
    Union,
)

import pytest
import spacy.tokenizer
import torch.nn
from confit import Config
from confit.utils.random import set_seed
from spacy.tokens import Doc, Span

from edsnlp.core.registries import registry
from edsnlp.data.converters import AttributesMappingArg, get_current_tokenizer
from edsnlp.metrics.dep_parsing import DependencyParsingMetric
from edsnlp.training.optimizer import LinearSchedule, ScheduledOptimizer
from edsnlp.training.trainer import GenericScorer, train
from edsnlp.utils.span_getters import SpanSetterArg, set_spans


@registry.factory.register("myproject.custom_dict2doc", spacy_compatible=False)
class CustomSampleGenerator:
    def __init__(
        self,
        *,
        tokenizer: Optional[spacy.tokenizer.Tokenizer] = None,
        name: str = "myproject.custom_dict2doc",
        span_setter: SpanSetterArg = {"ents": True, "*": True},
        bool_attributes: Union[str, Sequence[str]] = [],
        span_attributes: Optional[AttributesMappingArg] = None,
    ):
        self.tokenizer = tokenizer
        self.name = name
        self.span_setter = span_setter
        self.bool_attributes = bool_attributes
        self.span_attributes = span_attributes

    def __call__(self, obj):
        tok = get_current_tokenizer() if self.tokenizer is None else self.tokenizer
        doc = tok(obj["note_text"] or "")
        doc._.note_id = obj.get("note_id", obj.get("__FILENAME__"))
        doc._.note_datetime = obj.get("note_datetime")

        spans = []

        if self.span_attributes is not None:
            for dst in self.span_attributes.values():
                if not Span.has_extension(dst):
                    Span.set_extension(dst, default=None)

        for ent in obj.get("entities") or ():
            ent = dict(ent)
            span = doc.char_span(
                ent.pop("start"),
                ent.pop("end"),
                label=ent.pop("label"),
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
        for attr in self.bool_attributes:
            for span in spans:
                if span._.get(attr) is None:
                    span._.set(attr, False)
        return doc


def test_ner_qualif_train_diff_bert(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("ner_qlf_diff_bert_config.yml")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = Config.resolve(config["train"], registry=registry, root=config)
    nlp = train(**kwargs, output_dir=tmp_path, cpu=True)
    scorer = GenericScorer(**kwargs["scorer"])
    val_data = kwargs["val_data"]
    last_scores = scorer(nlp, val_data)

    # Check empty doc
    nlp("")

    assert last_scores["ner"]["micro"]["f"] > 0.4
    assert last_scores["qual"]["micro"]["f"] > 0.4


def test_ner_qualif_train_same_bert(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("ner_qlf_same_bert_config.yml")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = Config.resolve(config["train"], registry=registry, root=config)
    nlp = train(**kwargs, output_dir=tmp_path, cpu=True)
    scorer = GenericScorer(**kwargs["scorer"])
    val_data = kwargs["val_data"]
    last_scores = scorer(nlp, val_data)

    # Check empty doc
    nlp("")

    assert last_scores["ner"]["micro"]["f"] > 0.4
    assert last_scores["qual"]["micro"]["f"] > 0.4


def test_qualif_train(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("qlf_config.yml")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = Config.resolve(config["train"], registry=registry, root=config)
    nlp = train(**kwargs, output_dir=tmp_path, cpu=True)
    scorer = GenericScorer(**kwargs["scorer"])
    val_data = kwargs["val_data"]
    last_scores = scorer(nlp, val_data)

    # Check empty doc
    nlp("")

    assert last_scores["qual"]["micro"]["f"] >= 0.4


def test_dep_parser_train(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("dep_parser_config.yml")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = Config.resolve(config["train"], registry=registry, root=config)
    nlp = train(**kwargs, output_dir=tmp_path, cpu=True)
    scorer = GenericScorer(**kwargs["scorer"])
    val_data = list(kwargs["val_data"])
    last_scores = scorer(nlp, val_data)

    scorer_bis = GenericScorer(parser=DependencyParsingMetric(filter_expr="False"))
    # Just to test what happens if the scores indicate 2 roots
    val_data_bis = [Doc.from_docs([val_data[0], val_data[0]])]
    nlp.pipes.parser.decoding_mode = "mst"
    last_scores_bis = scorer_bis(nlp, val_data_bis)
    assert last_scores_bis["parser"]["uas"] == 0.0

    # Check empty doc
    nlp("")

    assert last_scores["dep"]["las"] >= 0.4


def test_optimizer():
    net = torch.nn.Linear(10, 10)
    optim = ScheduledOptimizer(
        torch.optim.AdamW,
        module=net,
        total_steps=10,
        groups={
            ".*": {
                "lr": 9e-4,
                "schedules": LinearSchedule(
                    warmup_rate=0.1,
                    start_value=0,
                ),
            }
        },
    )
    for param in net.parameters():
        assert "exp_avg" not in optim.optim.state[param]
    optim.initialize()
    for param in net.parameters():
        assert "exp_avg" in optim.optim.state[param]
    lr_values = [optim.optim.param_groups[0]["lr"]]
    for i in range(10):
        optim.step()
        lr_values.append(optim.optim.param_groups[0]["lr"])

    # close enough
    assert lr_values == pytest.approx(
        [
            0.0,
            0.0009,
            0.0008,
            0.0007,
            0.0006,
            0.0005,
            0.0004,
            0.0003,
            0.0002,
            0.0001,
            0.0,
        ]
    )
