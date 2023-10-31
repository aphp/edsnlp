import itertools
import json
import math
import random
import time
import typing
from collections import defaultdict
from collections.abc import Sized
from itertools import chain, repeat
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import spacy
import torch
from accelerate import Accelerator
from confit import Cli
from confit.registry import validate_arguments
from confit.utils.random import set_seed
from rich_logger import RichTablePrinter
from spacy.tokens import Doc
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
from edsnlp.utils.span_getters import SpanSetterArg, get_spans, set_spans

app = Cli(pretty_exceptions_show_locals=False)


LOGGER_FIELDS = {
    "step": {},
    "(.*)loss": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
    },
    "exact_ner/micro/(f|r|p)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"ner_\1",
    },
    "qualifier/micro/(f|r|p)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"qual_\1",
    },
    "lr": {"format": "{:.2e}"},
    "speed/(.*)": {"format": "{:.2f}", r"name": r"\1"},
    "labels": {"format": "{:.2f}"},
}


def flatten_dict(d, path=""):
    if not isinstance(d, dict):
        return {path: d}

    return {
        k: v
        for key, val in d.items()
        for k, v in flatten_dict(val, f"{path}/{key}" if path else key).items()
    }


class BatchSizeArg:
    """
    Batch size argument validator / caster for confit/pydantic

    Examples
    --------

    ```python
    def fn(batch_size: BatchSizeArg):
        return batch_size

    print(fn("10 samples"))
    # Out: (10, "samples")

    print(fn("10 words"))
    # Out: (10, "words")

    print(fn(10))
    # Out: (10, "samples")
    """

    @classmethod
    def validate(cls, value, config=None):
        value = str(value)
        parts = value.split()
        num = int(parts[0])
        if str(num) == parts[0]:
            if len(parts) == 1:
                return num, "samples"
            if parts[1] in ("words", "samples", "spans"):
                return num, parts[1]
        raise Exception(
            f"Invalid batch size: {value}, must be <int> samples|words|spans"
        )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


if typing.TYPE_CHECKING:
    BatchSizeArg = Tuple[int, str]  # noqa: F811


class LengthSortedBatchSampler:
    """
    Batch sampler that sorts the dataset by length and then batches
    sequences of similar length together. This is useful for transformer
    models that can then be padded more efficiently.

    Parameters
    ----------
    dataset: Iterable
        The dataset to sample from (can be a generator or a fixed size collection)
    batch_size: int
        The batch size
    batch_unit: str
        The unit of the batch size, either "words" or "samples"
    noise: int
        The amount of noise to add to the sequence length before sorting
        (uniformly sampled in [-noise, noise])
    drop_last: bool
        Whether to drop the last batch if it is smaller than the batch size
    buffer_size: Optional[int]
        The size of the buffer to use to shuffle the batches. If None, the buffer
        will be approximately the size of the dataset.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        batch_unit: str,
        noise=1,
        drop_last=True,
        buffer_size: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_unit = batch_unit
        self.noise = noise
        self.drop_last = drop_last
        self.buffer_size = buffer_size

    def __iter__(self):
        # Shuffle the dataset
        if self.batch_unit == "words":

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

        elif self.batch_unit == "spans":

            def sample_len(idx, noise=True):
                return len(
                    next(
                        v for k, v in self.dataset[idx].items() if k.endswith("begins")
                    )
                )

        else:

            def sample_len(idx, noise=True):
                return 1

        def make_batches():
            current_count = 0
            current_batch = []
            for idx in sorted_sequences:
                seq_size = sample_len(idx, noise=False)
                if current_count + seq_size > self.batch_size:
                    yield current_batch
                    current_batch = []
                    current_count = 0
                current_count += seq_size
                current_batch.append(idx)
            if len(current_batch):
                yield current_batch

        # Shuffle the batches in buffer that contain approximately
        # the full dataset to add more randomness
        if isinstance(self.dataset, Sized):
            total_count = sum(sample_len(i, False) for i in range(len(self.dataset)))

        assert (
            isinstance(self.dataset, Sized) or self.buffer_size is not None
        ), "Dataset must have a length or buffer_size must be specified"
        buffer_size = self.buffer_size or math.ceil(total_count / self.batch_size)

        # Sort sequences by length +- some noise
        sorted_sequences = chain.from_iterable(
            sorted(range(len(self.dataset)), key=sample_len) for _ in repeat(None)
        )

        # Batch sorted sequences
        batches = make_batches()
        buffers = batchify(batches, buffer_size)
        for buffer in buffers:
            random.shuffle(buffer)
            yield from buffer


class SubBatchCollater:
    """
    Collater that splits batches into sub-batches of a maximum size

    Parameters
    ----------
    nlp: Pipeline
        The pipeline object
    embedding: Transformer
        The transformer embedding pipe
    grad_accumulation_max_tokens: int
        The maximum number of tokens (word pieces) to accumulate in a single batch
    """

    def __init__(self, nlp, embedding, grad_accumulation_max_tokens):
        self.nlp = nlp
        self.embedding: Transformer = embedding
        self.grad_accumulation_max_tokens = grad_accumulation_max_tokens

    def __call__(self, seq):
        total = 0
        mini_batches = [[]]
        for sample_features in seq:
            num_tokens = sum(
                math.ceil(len(p) / self.embedding.stride) * self.embedding.window
                for key in sample_features
                if key.endswith("/input_ids")
                for p in sample_features[key]
            )
            if total + num_tokens > self.grad_accumulation_max_tokens:
                print(
                    f"Mini batch size was becoming too large: {total} > "
                    f"{self.grad_accumulation_max_tokens} so it was split"
                )
                total = 0
                mini_batches.append([])
            total += num_tokens
            mini_batches[-1].append(sample_features)
        return [self.nlp.collate(b) for b in mini_batches]


def subset_doc(doc: Doc, start: int, end: int) -> Doc:
    """
    Subset a doc given a start and end index.

    Parameters
    ----------
    doc: Doc
        The doc to subset
    start: int
        The start index
    end: int
        The end index

    Returns
    -------
    Doc
    """
    # TODO: review user_data copy strategy
    new_doc = doc[start:end].as_doc(copy_user_data=True)
    new_doc.user_data.update(doc.user_data)

    for name, group in doc.spans.items():
        new_doc.spans[name] = [
            spacy.tokens.Span(
                new_doc,
                max(0, span.start - start),
                min(end, span.end) - start,
                span.label,
            )
            for span in group
            if span.end > start and span.start < end
        ]

    return new_doc


def split_doc(
    doc: Doc,
    max_length: int = 0,
    randomize: bool = True,
    multi_sentence: bool = True,
) -> List[Doc]:
    """
    Split a doc into multiple docs of max_length tokens.

    Parameters
    ----------
    doc: Doc
        The doc to split
    max_length: int
        The maximum length of the resulting docs
    multi_sentence: bool
        Whether to split sentences across multiple docs
    randomize: bool
        Whether to randomize the split

    Returns
    -------
    Iterable[Doc]
    """
    if max_length <= 0:
        yield doc
    else:
        start = 0
        end = 0
        for ent in doc.ents:
            for token in ent:
                token.is_sent_start = False
        for sent in doc.sents if doc.has_annotation("SENT_START") else (doc[:],):
            # If the sentence adds too many tokens
            if sent.end - start > max_length:
                # But the current buffer too large
                while sent.end - start > max_length:
                    subset_end = start + int(
                        max_length * (random.random() ** 0.3 if randomize else 1)
                    )
                    yield subset_doc(doc, start, subset_end)
                    start = subset_end
                yield subset_doc(doc, start, sent.end)
                start = sent.end

            if not multi_sentence:
                yield subset_doc(doc, start, sent.end)
                start = sent.end

            # Otherwise, extend the current buffer
            end = sent.end

        yield subset_doc(doc, start, end)


@registry.misc.register("jsonl_dataset")
def jsonl_dataset(
    path,
    limit: Optional[int] = None,
    max_length: int = 0,
    randomize: bool = False,
    multi_sentence: bool = True,
    filter_expr: Optional[str] = None,
    span_setter: SpanSetterArg = {"ents": True, "gold-spans": True},
    note_id_field: str = "note_id",
    note_text_field: str = "note_text",
    entities_field: str = "entities",
):
    filter_fn = eval(f"lambda doc: {filter_expr}") if filter_expr else None
    assert not (
        limit is not None and isinstance(path, dict)
    ), "Cannot use specify both global limit and path-wise limit"
    if isinstance(path, (str, Path)):
        path = [path]
    if isinstance(path, list):
        path = {single_path: (limit or -1) for single_path in path}

    def load(nlp) -> List[Doc]:
        blank_nlp = edsnlp.Pipeline(nlp.lang, vocab=nlp.vocab)
        blank_nlp.add_pipe("eds.normalizer")
        blank_nlp.add_pipe("eds.sentences")

        # Load the jsonl data from path
        for single_path, limit in path.items():
            count = 0
            with open(single_path, "r") as f:
                lines = f
                if randomize:
                    lines = list(lines)
                    random.shuffle(lines)

                for line in lines:
                    if limit >= 0 and count >= limit:
                        break
                    raw = json.loads(line)
                    doc = nlp.make_doc(raw[note_text_field])
                    if note_id_field in raw:
                        doc._.note_id = raw[note_id_field]
                    doc = blank_nlp(doc)
                    if not (len(doc) and (filter_fn is None or filter_fn(doc))):
                        continue

                    count += 1

                    # Annotate entities from the raw data
                    set_spans(
                        doc,
                        [
                            doc.char_span(
                                ent["start"],
                                ent["end"],
                                label=ent["label"],
                                alignment_mode="expand",
                            )
                            for ent in raw[entities_field]
                        ],
                        span_setter,
                    )

                    for subdoc in split_doc(doc, max_length, randomize, multi_sentence):
                        if len(subdoc.text.strip()):
                            yield subdoc
                    else:
                        continue

                assert limit == 0 or count > 0, f"No data found in {single_path}"

    return load


@registry.misc.register("brat_dataset")
def brat_dataset(
    path,
    limit: Optional[int] = None,
    max_length: int = 0,
    randomize: bool = False,
    multi_sentence: bool = True,
    filter_expr: Optional[str] = None,
    span_setter: SpanSetterArg = {"ents": True, "gold-spans": True},
):
    filter_fn = eval(f"lambda doc: {filter_expr}") if filter_expr else None
    assert not (
        limit is not None and isinstance(path, dict)
    ), "Cannot use specify both global limit and path-wise limit"
    if isinstance(path, (str, Path)):
        path = [path]
    if isinstance(path, list):
        path = {single_path: (limit or -1) for single_path in path}

    def load(nlp) -> List[Doc]:
        blank_nlp = edsnlp.Pipeline(nlp.lang, vocab=nlp.vocab, vocab_config=None)
        blank_nlp.add_pipe("eds.normalizer")
        blank_nlp.add_pipe("eds.sentences")

        # Load the jsonl data from path
        for single_path, limit in path.items():
            count = 0
            docs = BratConnector(
                directory=single_path,
                span_setter=span_setter,
            ).brat2docs(blank_nlp, run_pipe=True)

            if randomize:
                docs = list(docs)
                random.shuffle(docs)

            for doc in docs:
                if 0 <= limit <= count:
                    break
                if not (len(doc) and (filter_fn is None or filter_fn(doc))):
                    continue
                count += 1

                for subdoc in split_doc(doc, max_length, randomize, multi_sentence):
                    if len(subdoc.text.strip()):
                        yield subdoc
                else:
                    continue

            assert limit == 0 or count > 0, f"No data found in {single_path}"

    return load


@validate_arguments
class GenericScorer:
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


@app.command(name="train", registry=registry)
def train(
    *,
    output_path: Path = Path("model"),
    nlp: Pipeline,
    train_data: Callable[[Pipeline], Iterable[Doc]],
    val_data: Callable[[Pipeline], Iterable[Doc]],
    seed: int = 42,
    data_seed: int = 42,
    max_steps: int = 1000,
    batch_size: BatchSizeArg = 2000,
    transformer_lr: float = 5e-5,
    task_lr: float = 3e-4,
    validation_interval: int = 10,
    max_grad_norm: float = 5.0,
    warmup_rate: float = 0.1,
    scorer: GenericScorer = GenericScorer(),
    grad_accumulation_max_tokens: int = 96 * 128,
    cpu: bool = False,
):
    trf_pipe = next(
        module
        for name, pipe in nlp.torch_components()
        for module_name, module in pipe.named_component_modules()
        if isinstance(module, Transformer)
    )

    set_seed(seed)
    with RichTablePrinter(LOGGER_FIELDS, auto_refresh=False) as logger:
        # Loading and adapting the training and validation data
        with set_seed(data_seed):
            train_docs: typing.List[spacy.tokens.Doc] = list(train_data(nlp))
            val_docs: typing.List[spacy.tokens.Doc] = list(val_data(nlp))

        # Initialize pipeline with training documents
        nlp.post_init(train_docs)

        # Preprocessing training data
        preprocessed = list(nlp.preprocess_many(train_docs, supervision=True))
        dataloader = torch.utils.data.DataLoader(
            preprocessed,
            batch_sampler=LengthSortedBatchSampler(
                preprocessed,
                batch_size=batch_size[0],
                batch_unit=batch_size[1],
            ),
            collate_fn=SubBatchCollater(
                nlp,
                trf_pipe,
                grad_accumulation_max_tokens=grad_accumulation_max_tokens,
            ),
        )
        trained_pipes = list(nlp.torch_components())
        print("Training", ", ".join([name for name, c in trained_pipes]))

        trf_params = set(trf_pipe.parameters())
        params = set(p for name, pipe in trained_pipes for p in pipe.parameters())
        optimizer = ScheduledOptimizer(
            torch.optim.AdamW(
                [
                    {
                        "params": list(params - trf_params),
                        "lr": task_lr,
                        "schedules": LinearSchedule(
                            total_steps=max_steps,
                            warmup_rate=warmup_rate,
                            start_value=task_lr,
                        ),
                    },
                    {
                        "params": list(trf_params),
                        "lr": transformer_lr,
                        "schedules": LinearSchedule(
                            total_steps=max_steps,
                            warmup_rate=warmup_rate,
                            start_value=0,
                        ),
                    },
                ][: 2 if transformer_lr else 1]
            )
        )
        grad_params = {p for group in optimizer.param_groups for p in group["params"]}
        print(
            "Optimizing:"
            + "".join(
                f"\n - {len(group['params'])} params "
                f"({sum(p.numel() for p in group['params'])} total)"
                for group in optimizer.param_groups
            )
        )
        print(f"Not optimizing {len(params - grad_params)} params")
        for param in params - grad_params:
            param.requires_grad_(False)

        accelerator = Accelerator(cpu=cpu)
        trained_pipes = [pipe for name, pipe in nlp.torch_components()]
        print("Device:", accelerator.device)
        [dataloader, optimizer, *trained_pipes] = accelerator.prepare(
            dataloader,
            optimizer,
            *trained_pipes,
        )

        cumulated_data = defaultdict(lambda: 0.0, count=0)

        iterator = itertools.chain.from_iterable(itertools.repeat(dataloader))
        all_metrics = []
        nlp.train(True)
        set_seed(seed)
        with tqdm(
            range(max_steps + 1),
            "Training model",
            leave=True,
            mininterval=5.0,
        ) as bar:
            for step in bar:
                if (step % validation_interval) == 0:
                    scores = scorer(nlp, val_docs)
                    all_metrics.append(
                        {
                            "step": step,
                            "lr": optimizer.param_groups[0]["lr"],
                            **cumulated_data,
                            **scores,
                        }
                    )
                    cumulated_data = defaultdict(lambda: 0.0, count=0)
                    nlp.to_disk(output_path)
                    (output_path / "train_metrics.json").write_text(
                        json.dumps(all_metrics, indent=2)
                    )
                    logger.log_metrics(flatten_dict(all_metrics[-1]))
                if step == max_steps:
                    break
                mini_batches = next(iterator)
                optimizer.zero_grad()
                for mini_batch in mini_batches:
                    loss = torch.zeros((), device=accelerator.device)
                    with nlp.cache():
                        for pipe in trained_pipes:
                            output = pipe.module_forward(mini_batch[pipe.name])
                            if "loss" in output:
                                loss += output["loss"]
                            for key, value in output.items():
                                if key.endswith("loss"):
                                    cumulated_data[key] += float(value)
                            if torch.isnan(loss):
                                raise ValueError(f"NaN loss at component {pipe.name}")

                    accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(grad_params, max_grad_norm)
                optimizer.step()

    return nlp


if __name__ == "__main__":
    app()
