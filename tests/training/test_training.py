import math
import random
import shutil
import time
from itertools import chain, count, islice, repeat
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from confit import Config
from confit.registry import validate_arguments
from confit.utils.random import set_seed
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


class LengthSortedBatchSampler:
    def __init__(self, dataset, batch_size, noise=1, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.noise = noise
        self.drop_last = drop_last

    def __iter__(self):
        # Shuffle the dataset
        def sample_len(idx):
            wt = next(
                v for k, v in self.dataset[idx].items() if k.endswith("word_tokens")
            )
            return len(wt) + random.randint(-self.noise, self.noise)

        # Sort sequences by length +- some noise
        sequences = chain.from_iterable(
            sorted(range(len(self.dataset)), key=sample_len) for _ in repeat(None)
        )

        # Batch sorted sequences
        batches = batchify(sequences, self.batch_size)

        # Shuffle the batches in buffer that contain approximately
        # the full dataset to add more randomness
        buffers = batchify(batches, math.ceil(len(self.dataset) / self.batch_size))
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
            # for doc in docs:
            #     for group in doc.spans:
            #         for span in doc.spans[group]:
            #             span._.negation = bool(span._.negation)
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
                ner_preds = list(nlp.pipe(clean_ner_docs))
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
                qlf_preds = list(nlp.pipe(clean_qlf_docs))
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
    batch_size: int = 4,
    lr: float = 8e-5,
    validation_interval: int = 10,
    device: str = "cpu",
    scorer: TestScorer = TestScorer(),
):
    import torch

    device = torch.device(device)
    set_seed(seed)

    # Loading and adapting the training and validation data
    train_docs = list(train_data(nlp))
    val_docs = list(val_data(nlp))

    # Taking the first `initialization_subset` samples to initialize the model
    nlp.post_init(iter(train_docs))  # iter just to show it's possible
    nlp.batch_size = batch_size

    # Preprocessing the training dataset into a dataloader
    preprocessed = list(nlp.preprocess_many(train_docs, supervision=True))
    dataloader = torch.utils.data.DataLoader(
        preprocessed,
        batch_sampler=LengthSortedBatchSampler(preprocessed, batch_size),
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
                            warmup_rate=0.1,
                            max_value=lr,
                            path="lr",
                        ),
                        # LinearSchedule(
                        #     total_steps=max_steps,
                        #     start_value=0.9,
                        #     max_value=0.9,
                        #     path="betas.0",
                        # ),
                    ],
                },
            ]
        )
    )
    print(
        "Number of optimized weight tensors", len(optimizer.param_groups[0]["params"])
    )

    # We will loop over the dataloader
    iterator = iter(dataloader)

    nlp.to(device)

    acc_loss = 0
    acc_steps = 0
    bar = tqdm(range(max_steps), "Training model", leave=True)
    for step in count():
        if step > 0 and (step % validation_interval) == 0 or step == max_steps:
            nlp.to_disk(output_path / "last-model")
            print(acc_loss / max(acc_steps, 1))
            acc_loss = 0
            acc_steps = 0
            last_scores = scorer(nlp, val_docs)
            print(last_scores, "lr", optimizer.param_groups[0]["lr"])
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
                # Test nan loss
                if torch.isnan(loss):
                    raise ValueError(f"NaN loss at component {name}")

        loss.backward()

        acc_loss += loss.item()
        acc_steps += 1

        optimizer.step()

        bar.update()

    optimizer.load_state_dict(optimizer.state_dict())

    assert Path(output_path / "last-model").exists()

    nlp = edsnlp.load(output_path / "last-model")

    return nlp


def test_ner_qualif_train(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("ner_qlf_config.cfg")
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = config["train"].resolve(registry=registry, root=config)
    nlp = train(**kwargs, output_path=tmp_path)
    scorer = TestScorer(**kwargs["scorer"])
    last_scores = scorer(nlp, kwargs["val_data"](nlp))

    assert last_scores["exact_ner"]["ents_f"] > 0.8


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


# def test_deft_train(run_in_test_dir, tmp_path):
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
    import tempfile

    tmp_path = tempfile.mkdtemp()
    print("TMP PATH", tmp_path)
    test_ner_qualif_train(None, tmp_path)
