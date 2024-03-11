import math
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import tokenizers
import torch
from confit import validate_arguments
from foldedtensor import as_folded_tensor
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig as BitsAndBytesConfig_
from typing_extensions import Literal, TypedDict

from edsnlp import Pipeline
from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent
from edsnlp.utils.span_getters import SpanGetterArg, get_spans

INITIAL_MAX_TOKENS_PER_DEVICE = 32 * 128
TransformerBatchInput = TypedDict(
    "TransformerBatchInput",
    {
        "input_ids": torch.LongTensor,
        "attention_mask": torch.BoolTensor,
        "token_window_indices": torch.LongTensor,
        "words_offsets": torch.LongTensor,
        "mask": torch.BoolTensor,
    },
)

BitsAndBytesConfig = validate_arguments(BitsAndBytesConfig_)


def compute_contextualization_scores(windows):
    ramp = torch.arange(0, windows.shape[1], 1)
    scores = (
        torch.min(ramp, windows.mask.sum(1, keepdim=True) - 1 - ramp)
        .clamp(min=0)
        .view(-1)
    )
    return scores


class Transformer(WordEmbeddingComponent[TransformerBatchInput]):
    """
    The `eds.transformer` component is a wrapper around HuggingFace's
    [transformers](https://huggingface.co/transformers/) library. If you are not
    familiar with transformers, a good way to start is the
    [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
    tutorial.

    Compared to using the raw Huggingface model, we offer a simple
    mechanism to split long documents into strided windows before feeding them to the
    model.

    Windowing
    ---------
    EDS-NLP's Transformer component splits long documents into smaller windows before
    feeding them to the model. This is done to avoid hitting the maximum number of
    tokens that can be processed by the model on a single device. The window size and
    stride can be configured using the `window` and `stride` parameters. The default
    values are 512 and 256 respectively, which means that the model will process windows
    of 512 tokens, each separated by 256 tokens. Whenever a token appears in multiple
    windows, the embedding of the "most contextualized" occurrence is used, i.e. the
    occurrence that is the closest to the center of its window.

    Here is an overview how this works to produce embeddings (shown in red) for each
    word of the document :

    <figure style="text-align: center" markdown>
    ![Transformer windowing](/assets/images/transformer-windowing.svg)
    </figure>

    Examples
    --------
    Here is an example of how to define a pipeline with a Transformer component:

    ```python
    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config=dict(
            model="prajjwal1/bert-tiny",
            window=128,
            stride=96,
        ),
    )
    ```

    You can then compose this embedding with a task specific component such as
    `eds.ner_crf`.

    Parameters
    ----------
    nlp: PipelineProtocol
        The pipeline instance
    name: str
        The component name
    model: str
        The Huggingface model name or path
    window: int
        The window size to use when splitting long documents into smaller windows
        before feeding them to the Transformer model (default: 512 = 512 - 2)
    stride: int
        The stride (distance between windows) to use when splitting long documents into
        smaller windows: (default: 96)
    max_tokens_per_device: Union[int, Literal["auto"]]
        The maximum number of tokens that can be processed by the model on a single
        device. This does not affect the results but can be used to reduce the memory
        usage of the model, at the cost of a longer processing time.

        If "auto", the component will try to estimate the maximum number of tokens that
        can be processed by the model on the current device at a given time.
    span_getter: Optional[SpanGetterArg]
        Which spans of the document should be embedded. Defaults to the full document
        if None.
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "eds.transformer",
        *,
        model: Union[str, Path],
        window: int = 128,
        stride: int = 96,
        max_tokens_per_device: Union[int, Literal["auto"]] = "auto",
        span_getter: Optional[SpanGetterArg] = None,
        new_tokens: Optional[List[Tuple[str, str]]] = [],
        quantization: Optional[BitsAndBytesConfig] = None,
        **kwargs,
    ):
        super().__init__(nlp, name)
        self.transformer = AutoModel.from_pretrained(
            model,
            add_pooling_layer=False,
            quantization_config=quantization,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.window = window
        self.stride = stride
        self.output_size = self.transformer.config.hidden_size
        self.empty_word_embedding = torch.nn.Parameter(
            torch.randn(
                (1, self.output_size),
            )
        )
        self.cfg = {}
        self.max_tokens_per_device = max_tokens_per_device
        self._mem_per_unit = None
        self.span_getter = span_getter

        if new_tokens:
            self.tokenizer.add_tokens(sorted(set(t[1] for t in new_tokens)))
            original_normalizer = self.tokenizer.backend_tokenizer.normalizer
            self.tokenizer.backend_tokenizer.normalizer = (
                tokenizers.normalizers.Sequence(
                    [
                        *(
                            tokenizers.normalizers.Replace(
                                tokenizers.Regex(pattern), replacement
                            )
                            for pattern, replacement in new_tokens
                        ),
                        original_normalizer,
                    ]
                )
            )
            # and add a new entry to the model's embeddings
            self.transformer.resize_token_embeddings(
                max(self.tokenizer.vocab.values()) + 1
            )

    def to_disk(self, path, *, exclude: Optional[Set[str]]):
        repr_id = object.__repr__(self)
        if repr_id in exclude:
            return
        self.tokenizer.save_pretrained(path)
        self.transformer.save_pretrained(path)
        for param in self.transformer.parameters():
            exclude.add(object.__repr__(param))
        cfg = super().to_disk(path, exclude=exclude) or {}
        cfg["model"] = f"./{path.as_posix()}"
        return cfg

    def preprocess(self, doc):
        res = {
            "input_ids": [],
            "word_tokens": [],
            "word_lengths": [],
        }

        for span in get_spans(doc, self.span_getter):
            # Preprocess it using LayoutLMv3
            prep = self.tokenizer(
                span.text,
                is_split_into_words=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )

            span_word_tokens, span_word_lengths = self.align_words_with_trf_tokens(
                span, prep.pop("offset_mapping")
            )
            res["input_ids"].append(prep["input_ids"])
            res["word_tokens"].append(span_word_tokens)
            res["word_lengths"].append(span_word_lengths)

        return res

    def collate(self, batch):
        # Flatten most of these arrays to process batches page per page and
        # not sample per sample

        S = self.stride
        W = self.window

        offset = 0
        window_max_size = 0
        window_count = 0
        windows = []
        windows_count_per_page = []
        for sample_input_ids in batch["input_ids"]:
            for span_input_ids in sample_input_ids:
                span_size = len(span_input_ids)
                num_span_windows = 1 + max(0, math.ceil((span_size - 2 - W) / S))
                windows.append(
                    [
                        [
                            offset + 0,
                            *range(
                                1 + offset + win_i * S,
                                1 + offset + min(win_i * S + W, span_size - 2),
                            ),
                            offset + len(span_input_ids) - 1,
                        ]
                        for win_i in range(0, num_span_windows)
                    ]
                )
                windows_count_per_page.append(len(windows[-1]))
                offset += len(span_input_ids)
                window_max_size = max(window_max_size, max(map(len, windows[-1])))
                window_count += len(windows[-1])

        windows = as_folded_tensor(
            windows,
            full_names=("sample", "window", "token"),
            data_dims=("window", "token"),
            dtype=torch.long,
        )
        indexer = torch.zeros(
            (0 if 0 in windows.shape else windows.max()) + 1, dtype=torch.long
        )

        # Sort each occurrence of an initial token by its contextualization score:
        # We can only use the amax reduction, so to retrieve the best occurrence, we
        # insert the index of the token output by the transformer inside the score
        # using a lexicographic approach
        # (score + index / n_tokens) ~ (score * n_tokens + index), taking the max,
        # and then retrieving the index of the token using the modulo operator.
        scores = compute_contextualization_scores(windows)
        scores = scores * len(scores) + torch.arange(len(scores))

        indexer.index_reduce_(
            dim=0,
            source=scores,
            index=windows.view(-1),
            reduce="amax",
        )
        indexer %= max(len(scores), 1)
        indexer = indexer.tolist()
        # Indexer: flattened, unpadded mapping tensor, that for each wordpiece in the
        # batch gives its position inside the flattened padded windows before/after the
        # transformer

        input_ids = as_folded_tensor(
            batch["input_ids"],
            data_dims=("subword",),
            full_names=("sample", "span", "subword"),
            dtype=torch.long,
        ).as_tensor()[windows]
        empty_id = input_ids.numel()

        word_indices = []
        word_offsets = []
        offset = 0
        indexer_offset = 0
        mask = []
        for sample_word_lengths, sample_word_tokens, sample_input_ids in zip(
            batch["word_lengths"], batch["word_tokens"], batch["input_ids"]
        ):
            for span_word_lengths, span_word_tokens, span_input_ids in zip(
                sample_word_lengths, sample_word_tokens, sample_input_ids
            ):
                offset = 0
                for length in span_word_lengths:
                    word_offsets.append(len(word_indices))
                    word_indices.extend(
                        [
                            indexer[indexer_offset + i] if i >= 0 else empty_id
                            for i in span_word_tokens[offset : offset + length]
                        ]
                    )
                    offset += length
                indexer_offset += len(span_input_ids)
                mask.append([True] * len(span_word_lengths))

        return {
            "input_ids": input_ids.as_tensor(),
            "input_mask": input_ids.mask,
            "word_indices": torch.as_tensor(word_indices),
            "word_offsets": torch.as_tensor(word_offsets),
            "mask": as_folded_tensor(
                mask,
                data_dims=(
                    "span",
                    "subword",
                ),
                full_names=("span", "subword"),
                dtype=torch.bool,
            ).as_tensor(),
        }

    def forward(self, batch):
        device = batch["input_ids"].device
        if len(batch["input_ids"]) == 0:
            return {
                "embeddings": torch.zeros(
                    (*batch["mask"].shape, self.output_size),
                    dtype=torch.float,
                    device=device,
                ),
                "mask": batch["mask"].clone(),
            }

        kwargs = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
        )
        auto_batch_size = device.type == "cuda" and self.max_tokens_per_device == "auto"
        trial_idx = 1
        while True:
            total_tokens = batch["input_ids"].numel()
            if auto_batch_size:  # pragma: no cover
                max_tokens = INITIAL_MAX_TOKENS_PER_DEVICE
                torch.cuda.synchronize(device)
                total_mem = torch.cuda.get_device_properties(device).total_memory
                allocated_mem = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                free_mem = total_mem - allocated_mem

                if self._mem_per_unit is not None:
                    max_tokens = int(free_mem // self._mem_per_unit)
            else:
                max_tokens = (
                    self.max_tokens_per_device
                    if self.max_tokens_per_device != "auto"
                    else INITIAL_MAX_TOKENS_PER_DEVICE
                )

            max_windows = max(1, max_tokens // batch["input_ids"].size(1))
            total_windows = batch["input_ids"].size(0)
            try:
                wordpiece_embeddings = [
                    self.transformer.base_model(
                        **{
                            k: None if v is None else v[offset : offset + max_windows]
                            for k, v in kwargs.items()
                        }
                    ).last_hidden_state
                    for offset in range(0, total_windows, max_windows)
                ]

                wordpiece_embeddings = (
                    torch.cat(wordpiece_embeddings, dim=0)
                    if len(wordpiece_embeddings) > 1
                    else wordpiece_embeddings[0]
                )

                if auto_batch_size:  # pragma: no cover
                    batch_mem = torch.cuda.max_memory_allocated(device)
                    current_mem_per_unit = batch_mem / min(total_tokens, max_tokens)
                    if self._mem_per_unit is None:
                        self._mem_per_unit = current_mem_per_unit
                    self._mem_per_unit = (
                        self._mem_per_unit * 0.7 + current_mem_per_unit * 0.3
                    )
                    torch.cuda.synchronize(device)  # Wait for all kernels to finish
            except RuntimeError as e:  # pragma: no cover
                if "out of memory" in str(e) and trial_idx <= 2:
                    print(
                        f"Out of memory: tried to fit {max_windows} "
                        f"in {free_mem / (1024 ** 3)} (try n°{trial_idx}/2)"
                    )
                    torch.cuda.empty_cache(device)
                    self._mem_per_unit = (free_mem / max_windows) * 1.5
                    trial_idx += 1
                    continue
                raise
            break

        mask = batch["mask"].clone()
        word_embeddings = torch.zeros(
            (mask.size(0), mask.size(1), wordpiece_embeddings.size(2)),
            dtype=torch.float,
            device=device,
        )
        embeddings_plus_empty = torch.cat(
            [
                wordpiece_embeddings.view(-1, wordpiece_embeddings.size(2)),
                self.empty_word_embedding,
            ],
            dim=0,
        )
        word_embeddings[mask] = torch.nn.functional.embedding_bag(
            input=batch["word_indices"],
            weight=embeddings_plus_empty,
            offsets=batch["word_offsets"],
        )
        return {
            "embeddings": word_embeddings,
            "mask": mask,
        }

    def align_words_with_trf_tokens(self, doc, trf_char_indices):
        token_i = 0
        n_trf_tokens = len(trf_char_indices)
        word_tokens = []
        word_lengths = [0] * (len(doc))
        doc_start = doc[0].idx if len(doc) else 0

        for word_i, word in enumerate(doc):
            length = 0
            word_begin = word.idx - doc_start
            word_end = word_begin + len(word)

            for j in range(token_i, n_trf_tokens):
                if trf_char_indices[j][1] <= word_begin:
                    token_i += 1
                elif trf_char_indices[j][0] >= word_end:
                    break
                else:
                    length += 1
                    word_tokens.append(j)
            if length > 0:
                word_lengths[word_i] = length
            else:
                word_tokens.append(-1)
                word_lengths[word_i] = 1

        return word_tokens, word_lengths
