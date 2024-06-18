# ruff: noqa: E402

import pytest

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
from spacy.tokens import Span

from edsnlp.core.registries import registry
from edsnlp.data.converters import AttributesMappingArg, get_current_tokenizer
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.train import GenericScorer, Reader, train
from edsnlp.utils.span_getters import SpanSetterArg, set_spans


@registry.factory.register("myproject.custom_dict2doc", spacy_compatible=False)
class PseudoDictReader:
    def __init__(
        self,
        *,
        tokenizer: Optional[spacy.tokenizer.Tokenizer] = None,
        name: str = "eds_pseudo.read_dict",
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


def test_ner_qualif_train(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("ner_qlf_config.cfg")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = config["train"].resolve(registry=registry, root=config)
    nlp = train(**kwargs, output_path=tmp_path, cpu=True)
    scorer = GenericScorer(**kwargs["scorer"])
    last_scores = scorer(nlp, Reader(**kwargs["val_data"])(nlp))

    # Check empty doc
    nlp("")

    assert last_scores["ner"]["micro"]["f"] > 0.4
    assert last_scores["qual"]["micro"]["f"] > 0.4


def test_qualif_train(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("qlf_config.cfg")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = config["train"].resolve(registry=registry, root=config)
    nlp = train(**kwargs, output_path=tmp_path, cpu=True)
    scorer = GenericScorer(**kwargs["scorer"])
    last_scores = scorer(nlp, Reader(**kwargs["val_data"])(nlp))

    # Check empty doc
    nlp("")

    assert last_scores["qual"]["micro"]["f"] >= 0.4


def test_optimizer():
    net = torch.nn.Linear(10, 10)
    optim = ScheduledOptimizer(
        torch.optim.AdamW(
            [
                {
                    "params": list(net.parameters()),
                    "lr": 9e-4,
                    "schedules": LinearSchedule(
                        total_steps=10,
                        warmup_rate=0.1,
                        start_value=0,
                    ),
                }
            ]
        )
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
