import math
import random
import shutil
from collections import defaultdict
from itertools import chain, count, repeat
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import torch
from confit import Config
from confit.registry import validate_arguments
from confit.utils.random import set_seed
from spacy.tokens import Doc, Span
from torch.utils.data import DataLoader
from tqdm import tqdm

from edsnlp.connectors.brat import BratConnector
from edsnlp.core.pipeline import Pipeline
from edsnlp.core.registry import registry
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.utils.collections import batchify
from edsnlp.utils.filter import filter_spans


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


@registry.misc.register("deft_span_getter")
def make_span_getter():
    def span_getter(doclike: Union[Doc, Span]) -> List[Span]:
        """
        Get the spans of a span group that are contained inside a doclike object.
        Parameters
        ----------
        doclike : Union[Doc, Span]
            Doclike object to act as a mask.
        group : str
            Group name from which to get the spans.
        Returns
        -------
        List[Span]
            List of spans.
        """
        if isinstance(doclike, Doc):
            # return [
            #     ent
            #     for group in doclike.doc.spans
            #     for ent in doclike.spans.get(group, ())
            # ]
            return doclike.ents
        else:
            # return [
            #     span
            #     for group in doclike.doc.spans
            #     for span in doclike.doc.spans.get(group, ())
            #     if span.start >= doclike.start and span.end <= doclike.end
            # ]
            return doclike.ents

    return span_getter


@registry.misc.register("brat_dataset")
def brat_dataset(path, limit: Optional[int] = None, span_getter=make_span_getter()):
    def load(nlp):
        raw_data = BratConnector(path).load_brat()
        assert len(raw_data) > 0, "No data found in {}".format(path)
        if limit is not None:
            raw_data = raw_data[:limit]

        # Initialize the docs (tokenize them)
        docs: List[Doc] = [nlp.make_doc(raw["text"]) for raw in raw_data]

        normalizer = nlp.get_pipe("normalizer")
        docs = [normalizer(doc) for doc in docs]

        sentencizer = nlp.get_pipe("sentencizer")
        docs = [sentencizer(doc) for doc in docs]

        # Annotate entities from the raw data
        for doc, raw in zip(docs, raw_data):
            ents = []
            span_groups = defaultdict(list)
            for ent in raw["entities"]:
                span = doc.char_span(
                    ent["fragments"][0]["begin"],
                    ent["fragments"][-1]["end"],
                    label=ent["label"],
                    alignment_mode="expand",
                )
                ents.append(span)
                span_groups[ent["label"]].append(span)
            doc.ents = filter_spans(ents)
            doc.spans.update(span_groups)

        new_docs = []
        for doc in docs:
            for sent in doc.sents:
                if len(span_getter(sent)):
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
        return new_docs

    return load


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
):
    device = torch.device(device)
    set_seed(seed)

    # Loading and adapting the training and validation data
    train_docs = list(train_data(nlp))
    val_docs = list(val_data(nlp))

    # Taking the first `initialization_subset` samples to initialize the model

    nlp.post_init(iter(train_docs))  # iter just to show it's possible
    nlp.batch_size = batch_size

    # assert nlp.get_pipe('ner').embedding is nlp.get_pipe('embedding')

    # print(nlp.config.to_str())

    # Preprocessing the training dataset into a dataloader
    preprocessed = list(nlp.preprocess_many(train_docs, supervision=True))
    dataloader = DataLoader(
        preprocessed,
        batch_sampler=LengthSortedBatchSampler(preprocessed, batch_size),
        collate_fn=nlp.collate,
    )

    trf_params = set(nlp.get_pipe("ner").embedding.embedding.parameters())
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
                            warmup_rate=0.5,
                            start_value=lr,
                            path="lr",
                        ),
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
    print(
        "Number of optimized weight tensors", len(optimizer.param_groups[0]["params"])
    )

    # We will loop over the dataloader
    iterator = iter(dataloader)

    # for name, pipe in nlp.torch_components():
    #    pipe.train(False)

    nlp.to(device)

    acc_loss = 0
    acc_steps = 0
    bar = tqdm(range(max_steps + 1), "Training model", leave=True)
    for step in count():
        if (step % validation_interval) == 0 or step == max_steps:
            nlp.to_disk(output_path / "last-model")
            print(acc_loss / max(acc_steps, 1))
            acc_loss = 0
            acc_steps = 0
            last_scores = nlp.score(val_docs)
            print(last_scores, "lr", optimizer.param_groups[0]["lr"])
        if step == max_steps:
            break
        batch = next(iterator)
        n_words = batch["ner"]["embedding"]["mask"].sum().item()
        n_padded = torch.numel(batch["ner"]["embedding"]["mask"])
        n_words_bert = batch["ner"]["embedding"]["attention_mask"].sum().item()
        n_padded_bert = torch.numel(batch["ner"]["embedding"]["attention_mask"])
        bar.set_postfix(
            n_words=n_words,
            ratio=n_words / n_padded,
            n_wp=n_words_bert,
            bert_ratio=n_words_bert / n_padded_bert,
        )
        optimizer.zero_grad()

        loss = torch.zeros((), device=device)
        with nlp.cache():
            for name, component in nlp.torch_components():
                output = component.module_forward(
                    batch[name],
                )
                loss += output.get("loss", 0)

        loss.backward()

        acc_loss += loss.item()
        acc_steps += 1

        optimizer.step()

        bar.update()

    optimizer.load_state_dict(optimizer.state_dict())

    assert Path(output_path / "last-model").exists()

    nlp.from_disk(output_path / "last-model")

    list(nlp.pipe(val_data(nlp)))

    assert last_scores["ner"]["ents_f"] == 1.0


def test_train(run_in_test_dir, tmp_path):
    set_seed(42)
    config = Config.from_disk("config.cfg")
    shutil.rmtree(tmp_path, ignore_errors=True)
    print(config.to_str())
    train(
        **config["train"].resolve(registry=registry, root=config),
        output_path=tmp_path,
    )
