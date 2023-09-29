import math
import random
import shutil
import time
import typing
from itertools import chain, count, islice, repeat
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

from confit import Config
from confit.registry import validate_arguments
from confit.utils.random import set_seed
from rich_logger import RichTablePrinter
from spacy.tokens import Doc, Span
from tqdm import tqdm

import edsnlp
from edsnlp.connectors.brat import BratConnector
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registry import registry
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.pipelines.trainable.embeddings.transformer.transformer import Transformer
from edsnlp.scorers import Scorer
from edsnlp.utils.bindings import BINDING_SETTERS
from edsnlp.utils.collections import batchify
from edsnlp.utils.span_getters import SpanGetterArg, get_spans

LOGGER_FIELDS = {
    "step": {},
    "(.*)loss": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
    },
    "exact_ner/ents_(f|r|p)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"ner_\1",
    },
    "qualifier/qual_(f|r|p)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"qual_\1",
    },
    "lr": {"format": "{:.2e}"},
    "speed/(.*)": {"format": "{:.2f}", r"name": r"\1"},
    "labels": {"format": "{:.2f}"},
}


def flatten_dict(root, depth=-1):
    res = {}

    def rec(d, path, current_depth):
        for k, v in d.items():
            if isinstance(v, dict) and current_depth != depth:
                rec(v, path + "/" + k if path is not None else k, current_depth + 1)
            else:
                res[path + "/" + k if path is not None else k] = v

    rec(root, None, 0)
    return res


class BatchSizeArg:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @classmethod
    def validate(cls, value, config=None):
        value = str(value)
        parts = value.split()
        num = int(parts[0])
        if str(num) == parts[0]:
            if len(parts) == 1:
                return num, "samples"
            if parts[1] in ("words", "samples"):
                return num, parts[1]
        raise Exception(f"Invalid batch size: {value}, must be <int> samples|words")

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


if typing.TYPE_CHECKING:
    BatchSizeArg = Tuple[int, str]  # noqa: F811


class LengthSortedBatchSampler:
    def __init__(
        self, dataset, batch_size: int, batch_unit: str, noise=1, drop_last=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_unit = batch_unit
        self.noise = noise
        self.drop_last = drop_last

    def __iter__(self):
        # Shuffle the dataset
        def sample_len(idx, noise=True):
            count = sum(
                len(x)
                for x in next(
                    v
                    for k, v in self.dataset[idx].items()
                    if k.endswith("word_lengths")
                )
            )
            if not noise:
                return count
            return count + random.randint(-self.noise, self.noise)

        def make_batches():
            current_count = 0
            current_batch = []
            for idx in sorted_sequences:
                if self.batch_unit == "words":
                    seq_size = sample_len(idx, noise=False)
                    if current_count + seq_size > self.batch_size:
                        yield current_batch
                        current_batch = []
                        current_count = 0
                    current_count += seq_size
                    current_batch.append(idx)
                else:
                    if len(current_batch) == self.batch_size:
                        yield current_batch
                        current_batch = []
                    current_batch.append(idx)
            if len(current_batch):
                yield current_batch

        # Sort sequences by length +- some noise
        sorted_sequences = chain.from_iterable(
            sorted(range(len(self.dataset)), key=sample_len) for _ in repeat(None)
        )

        # Batch sorted sequences
        batches = make_batches()

        # Shuffle the batches in buffer that contain approximately
        # the full dataset to add more randomness
        if self.batch_unit == "words":
            total_count = sum(
                sample_len(idx, noise=False) for idx in range(len(self.dataset))
            )
        else:
            total_count = len(self.dataset)
        buffers = batchify(batches, math.ceil(total_count / self.batch_size))
        for buffer in buffers:
            random.shuffle(buffer)
            yield from buffer


@registry.misc.register("brat_dataset")
def brat_dataset(path, limit: Optional[int] = None, split_docs: bool = False):
    def load(nlp):
        with nlp.select_pipes(enable=["normalizer", "sentencizer"]):
            docs = list(
                islice(
                    BratConnector(path).brat2docs(nlp, run_pipe=True),
                    limit,
                )
            )
            if Span.has_extension("negation"):
                for doc in docs:
                    for group in doc.spans:
                        for span in doc.spans[group]:
                            span._.negation = bool(span._.negation)
        assert len(docs) > 0, "No data found in {}".format(path)

        new_docs = []
        for doc in docs:
            if split_docs:
                for sent in doc.sents:
                    new_doc = sent.as_doc(copy_user_data=True)
                    for group in doc.spans:
                        new_doc.spans[group] = [
                            Span(
                                new_doc,
                                span.start - sent.start,
                                span.end - sent.start,
                                span.label_,
                            )
                            for span in doc.spans.get(group, ())
                            if span.start >= sent.start and span.end <= sent.end
                        ]
                    new_docs.append(new_doc)
            else:
                new_docs.append(doc)
        new_docs = new_docs[:limit]
        return new_docs

    return load


@validate_arguments
class TestScorer:
    def __init__(
        self,
        ner: Dict[str, Scorer] = {},
        qualifier: Dict[str, Scorer] = {},
    ):
        self.ner_scorers = ner
        self.qlf_scorers = qualifier

    def __call__(self, nlp: Pipeline, docs):
        scores = {}

        # Speed
        t0 = time.time()
        list(nlp.pipe(d.text for d in tqdm(docs, desc="Computing model speed")))
        duration = time.time() - t0
        scores["speed"] = dict(
            wps=sum(len(d) for d in docs) / duration,
            dps=len(docs) / duration,
        )

        # NER
        if nlp.has_pipe("ner"):
            clean_ner_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
            for d in clean_ner_docs:
                d.ents = []
                d.spans.clear()
            with nlp.select_pipes(enable=["ner"]):
                ner_preds = list(nlp.pipe(tqdm(clean_ner_docs, desc="Predicting")))
            for name, scorer in self.ner_scorers.items():
                scores[name] = scorer(docs, ner_preds)

        # Qualification
        if nlp.has_pipe("qualifier"):
            qlf_pipe = nlp.get_pipe("qualifier")
            clean_qlf_docs = [d.copy() for d in tqdm(docs, desc="Copying docs")]
            for doc in clean_qlf_docs:
                for span in get_spans(doc, qlf_pipe.span_getter):
                    for qlf in qlf_pipe.qualifiers:
                        BINDING_SETTERS[(qlf, None)](span)
            with nlp.select_pipes(enable=["qualifier"]):
                qlf_preds = list(nlp.pipe(tqdm(clean_qlf_docs, desc="Predicting")))
            for name, scorer in self.qlf_scorers.items():
                scores[name] = scorer(docs, qlf_preds)
        return scores


@validate_arguments
def train(
    output_path: Path,
    nlp: Pipeline,
    train_data: Callable[[Pipeline], Iterable[Doc]],
    val_data: Callable[[Pipeline], Iterable[Doc]],
    seed: int = 42,
    max_steps: int = 1000,
    batch_size: BatchSizeArg = 4,
    lr: float = 8e-5,
    validation_interval: int = 10,
    warmup_rate: float = 0.1,
    device: str = "cpu",
    scorer: TestScorer = TestScorer(),
):
    import torch

    with RichTablePrinter(LOGGER_FIELDS, auto_refresh=False) as logger:

        device = torch.device(device)
        set_seed(seed)

        # Loading and adapting the training and validation data
        train_docs = list(train_data(nlp))
        val_docs = list(val_data(nlp))

        # Taking the first `initialization_subset` samples to initialize the model
        nlp.post_init(iter(train_docs))  # iter just to show it's possible

        # Preprocessing the training dataset into a dataloader
        preprocessed = list(nlp.preprocess_many(train_docs, supervision=True))
        dataloader = torch.utils.data.DataLoader(
            preprocessed,
            batch_sampler=LengthSortedBatchSampler(
                preprocessed,
                batch_size=batch_size[0],
                batch_unit=batch_size[1],
            ),
            collate_fn=nlp.collate,
        )

        trf_params = set(
            next(
                module
                for name, pipe in nlp.torch_components()
                for module_name, module in pipe.named_component_modules()
                if isinstance(module, Transformer)
            ).parameters()
        )
        for param in trf_params:
            param.requires_grad = False
        optimizer = ScheduledOptimizer(
            torch.optim.AdamW(
                [
                    {
                        "params": list(set(nlp.parameters()) - trf_params),
                        "lr": lr,
                        "schedules": [
                            LinearSchedule(
                                total_steps=max_steps,
                                warmup_rate=warmup_rate,
                                max_value=lr,
                                path="lr",
                            ),
                            # This is just to test deep schedules
                            LinearSchedule(
                                total_steps=max_steps,
                                start_value=0.9,
                                max_value=0.9,
                                path="betas.0",
                            ),
                        ],
                    },
                ]
            )
        )
        trainable_params = set(p for g in optimizer.param_groups for p in g["params"])
        print("Trainable params", sum(p.numel() for p in trainable_params))
        print(
            "Non-trainable params",
            sum(p.numel() for p in nlp.parameters() if p not in trainable_params),
        )

        # We will loop over the dataloader
        iterator = iter(dataloader)

        nlp.to(device)

        losses = {name: 0 for name, _ in nlp.torch_components()}
        acc_steps = 0
        bar = tqdm(range(max_steps), "Training model", leave=True)

        for step in count():
            if step > 0 and (step % validation_interval) == 0 or step == max_steps:
                nlp.to_disk(output_path)
                scores = scorer(nlp, val_docs)
                metrics = flatten_dict(
                    {
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": sum(losses.values()) / max(acc_steps, 1),
                        **{
                            k + "_loss": v / max(acc_steps, 1)
                            for k, v in losses.items()
                        },
                        **scores,
                    }
                )
                logger.log_metrics(metrics)
                losses = {name: 0 for name, _ in nlp.torch_components()}
                acc_steps = 0
            if step == max_steps:
                break
            batch = next(iterator)
            optimizer.zero_grad()

            loss = torch.zeros((), device=device)
            with nlp.cache():
                for name, component in nlp.torch_components():
                    output = component.module_forward(
                        batch[name],
                    )
                    loss += output.get("loss", 0)
                    losses[name] += float(output.get("loss", 0))
                    # Test nan loss
                    if torch.isnan(loss):
                        raise ValueError(f"NaN loss at component {name}")

            loss.backward()

            # Max grad norm 5
            torch.nn.utils.clip_grad_norm_(trainable_params, 5)

            acc_steps += 1

            optimizer.step()

            bar.update()

    optimizer.load_state_dict(optimizer.state_dict())

    assert Path(output_path).exists()

    nlp = edsnlp.load(output_path)

    return nlp


def test_ner_qualif_train(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("ner_qlf_config.cfg")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = config["train"].resolve(registry=registry, root=config)
    nlp = train(**kwargs, output_path=tmp_path)
    scorer = TestScorer(**kwargs["scorer"])
    last_scores = scorer(nlp, kwargs["val_data"](nlp))

    assert last_scores["exact_ner"]["ents_f"] > 0.5
    assert last_scores["qualifier"]["qual_f"] > 0.5


@registry.misc.register("sentence_span_getter")
def make_sentence_span_getter(span_getter: SpanGetterArg):
    def get_sentence_spans(doc: Doc):
        return list(dict.fromkeys(ent.sent for ent in get_spans(doc, span_getter)))

    return get_sentence_spans


def test_qualif_train(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("qlf_config.cfg")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = config["train"].resolve(registry=registry, root=config)
    nlp = train(**kwargs, output_path=tmp_path)
    scorer = TestScorer(**kwargs["scorer"])
    last_scores = scorer(nlp, kwargs["val_data"](nlp))

    assert last_scores["qualifier"]["qual_f"] > 0.5


# def deft_train(run_in_test_dir, tmp_path):
#     set_seed(42)
#     config = Config.from_disk("deft_config.cfg")
#     shutil.rmtree(tmp_path, ignore_errors=True)
#     kwargs = config["train"].resolve(registry=registry, root=config)
#     nlp = train(**kwargs, output_path=tmp_path)
#     scorer = TestScorer(**kwargs["scorer"])
#     last_scores = scorer(nlp, kwargs["val_data"](nlp))
#
#     assert last_scores["qualifier"]["qual_f"] > 0.5


if __name__ == "__main__":

    tmp_path = Path("artifact/model-last")
    print("TMP PATH", tmp_path)
    test_qualif_train(None, tmp_path)
