import warnings
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import foldedtensor as ft
import tokenizers
import tokenizers.normalizers
import torch
from confit import VisibleDeprecationWarning, validate_arguments
from transformers import AutoModel, AutoTokenizer
from typing_extensions import Literal, TypedDict

from edsnlp import Pipeline
from edsnlp.core.torch_component import cached
from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent
from edsnlp.utils.span_getters import SpanGetterArg

try:
    from transformers import BitsAndBytesConfig as BitsAndBytesConfig_

    BitsAndBytesConfig = validate_arguments(BitsAndBytesConfig_)
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None

INITIAL_MAX_TOKENS_PER_DEVICE = 32 * 128
TransformerBatchInput = TypedDict(
    "TransformerBatchInput",
    {
        "input_ids": ft.FoldedTensor,
        "word_indices": torch.Tensor,
        "word_offsets": ft.FoldedTensor,
        "empty_word_indices": torch.Tensor,
    },
)
"""
input_ids: FoldedTensor
    Tokenized input (prompt + text) to embed
word_indices: torch.LongTensor
    Flattened indices of the word's wordpieces in the flattened input_ids
word_offsets: FoldedTensor
    Offsets of the word's wordpieces in the flattened input_ids
empty_word_indices: torch.LongTensor
    Indices of empty words in the flattened input_ids
"""

TransformerBatchOutput = TypedDict(
    "TransformerBatchOutput",
    {
        "embeddings": ft.FoldedTensor,
    },
)
"""
embeddings: FoldedTensor
    The embeddings of the words
"""


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
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        eds.transformer(
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
    training_stride: bool
        If False, the stride will be set to the window size during training, meaning
        that there will be no overlap between windows. If True, the stride will be set
        to the `stride` parameter during training, just like during inference.
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
        name: str = "transformer",
        *,
        model: Union[str, Path],
        window: int = 128,
        stride: int = 96,
        training_stride: bool = True,
        max_tokens_per_device: Union[int, Literal["auto"]] = "auto",
        span_getter: Optional[SpanGetterArg] = None,
        new_tokens: Optional[List[Tuple[str, str]]] = [],
        quantization: Optional[BitsAndBytesConfig] = None,
        **kwargs,
    ):
        super().__init__(nlp, name)

        if span_getter is not None:
            warnings.warn(
                "The `span_getter` parameter of the `eds.transformer` component is "
                "deprecated. Please use the `context_getter` parameter of the "
                "other higher level task components instead.",
                VisibleDeprecationWarning,
            )

        kwargs = dict(kwargs)

        if quantization is not None:
            kwargs["quantization_config"] = quantization

        self.transformer = AutoModel.from_pretrained(model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.window = window
        self.stride = stride
        self.training_stride = training_stride
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
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        if self.cls_token_id is None:
            [self.cls_token_id] = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.special_tokens_map["bos_token"]]
            )
        if self.sep_token_id is None:
            [self.sep_token_id] = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.special_tokens_map["eos_token"]]
            )

        if new_tokens:
            self.tokenizer.add_tokens(sorted(set(t[1] for t in new_tokens)))
            original_normalizer = self.tokenizer.backend_tokenizer.normalizer
            self.tokenizer.backend_tokenizer.normalizer = (
                tokenizers.normalizers.Sequence(
                    [  # type: ignore
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

        # Fix for https://github.com/aphp/edsnlp/issues/317
        old_params_data = {}
        for param in self.transformer.parameters():
            if not param.is_contiguous():
                old_params_data[param] = param.data
                param.data = param.data.contiguous()

        self.transformer.save_pretrained(path)

        # Restore non-contiguous tensors
        for param, data in old_params_data.items():
            param.data = data

        for param in self.transformer.parameters():
            exclude.add(object.__repr__(param))
        cfg = super().to_disk(path, exclude=exclude) or {}
        cfg["model"] = f"./{path.as_posix()}"
        return cfg

    @cached(
        key=lambda self, doc, *, contexts=None, prompts=(): (
            (
                hash(doc),
                tuple(hash(c) for c in ([doc[:]] if contexts is None else contexts)),
                tuple(prompts),
            )
        )
    )
    def preprocess(self, doc, *, contexts=None, prompts=()):
        res = {
            "input_ids": [],
            "word_tokens": [],
            "word_lengths": [],
            "prompts": [],
            "stats": {"tokens": 0, "words": 0, "contexts": 0},
        }

        # Tokenize prompts
        prompts_input_ids = [
            self.tokenizer(
                prompt,
                is_split_into_words=False,
                add_special_tokens=False,
                return_attention_mask=False,
                return_offsets_mapping=False,
            ).input_ids
            for prompt in prompts
        ] or [[]]
        if contexts is None:
            contexts = [doc[:]] * len(prompts_input_ids)
        elif not prompts:
            prompts_input_ids = [[]] * len(contexts)
        else:
            assert len(contexts) == len(prompts_input_ids), (
                "The number of spans and prompts passed in the `preprocess` "
                "method should be the same."
            )

        for ctx, prompt in zip(contexts, prompts_input_ids):
            prep = self.tokenizer(
                ctx.text,
                is_split_into_words=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False,
            )

            span_word_tokens, span_word_lengths = self.align_words_with_trf_tokens(
                ctx, prep.pop("offset_mapping")
            )
            res["input_ids"].append(prep["input_ids"])
            res["word_tokens"].append(span_word_tokens)
            res["word_lengths"].append(span_word_lengths)
            res["prompts"].append(prompt)

            res["stats"]["tokens"] += len(prep["input_ids"])
            res["stats"]["words"] += len(span_word_lengths)
            res["stats"]["contexts"] += 1

        return res

    def collate(self, batch):
        """
        How this works:
        1. Iterate over samples, and in each sample over spans of text to embed
           independently, and extract their input ids (and optionally prompts)
           in a list `input_ids` that will be passed to the transformer.

           `embeds = self.embedding(input_ids)`
        2. Since we want to aggregate over words, and have overlapping spans, we need
           to process indices carefully. Once the individual spans are embedded, we
           will flatten them...

           `flat_embeds = embeds.view(-1, embeds.size(2))[indexer]`

        Parameters
        ----------
        batch

        Returns
        -------

        """
        stride = (
            self.window if self.training and not self.training_stride else self.stride
        )
        max_seq_size = max(
            [
                2  # CLS and SEP tokens
                + (  # Prompt tokens (prompt + [SEP])
                    len(span_prompt_input_ids) + 1 if span_prompt_input_ids else 0
                )
                + min(self.window, len(span_text_input_ids))  # Text tokens
                for sample_text_input_ids, sample_prompt_input_ids in zip(
                    batch["input_ids"],
                    batch["prompts"],
                )
                for span_text_input_ids, span_prompt_input_ids in zip(
                    sample_text_input_ids,
                    sample_prompt_input_ids,
                )
            ]
            or [0]
        )
        input_ids = []
        token_indices = []
        word_indices = []
        word_offsets = []
        empty_word_indices = []
        overlap = self.window - stride
        word_offset = 0
        all_word_wp_offset = 0
        for (
            sample_text_input_ids,
            sample_prompt_input_ids,
            sample_word_lengths,
            sample_word_tokens,
        ) in zip(
            batch["input_ids"],
            batch["prompts"],
            batch["word_lengths"],
            batch["word_tokens"],
        ):
            sample_word_offsets = []
            word_offsets.append(sample_word_offsets)
            for (
                span_text_input_ids,
                span_prompt_input_ids,
                span_word_lengths,
                span_word_tokens,
            ) in zip(
                sample_text_input_ids,
                sample_prompt_input_ids,
                sample_word_lengths,
                sample_word_tokens,
            ):
                prompt_input_ids = [self.cls_token_id]
                if span_prompt_input_ids:
                    prompt_input_ids.extend([*span_prompt_input_ids, self.sep_token_id])
                windows_offsets = list(
                    range(0, max(len(span_text_input_ids) - overlap, 1), stride)
                )
                span_token_indices = []
                for idx, offset in enumerate(windows_offsets):
                    total_offset = len(input_ids) * max_seq_size + len(prompt_input_ids)
                    window_text_input_ids = span_text_input_ids[
                        offset : offset + self.window
                    ]
                    window_input_ids = (
                        prompt_input_ids + window_text_input_ids + [self.sep_token_id]
                    )
                    left_overlap = overlap // 2 if offset > 0 else 0
                    right_overlap = (
                        (overlap + 1) // 2 if idx < len(windows_offsets) - 1 else 0
                    )
                    wp_indices = list(
                        range(
                            total_offset + left_overlap,
                            total_offset + len(window_text_input_ids) - right_overlap,
                        )
                    )
                    span_token_indices.extend(wp_indices)
                    input_ids.append(window_input_ids)

                token_indices.append(span_token_indices)

                span_word_wp_offsets = []
                sample_word_offsets.append(span_word_wp_offsets)
                word_wp_offset = 0
                for length in span_word_lengths:
                    if length == 0:
                        empty_word_indices.append(word_offset)
                    span_word_wp_offsets.append(all_word_wp_offset + word_wp_offset)
                    word_wp_indices = [
                        span_token_indices[i]
                        for i in span_word_tokens[
                            word_wp_offset : word_wp_offset + length
                        ]
                    ]
                    word_indices.extend(word_wp_indices)
                    word_wp_offset += length
                    word_offset += 1
                all_word_wp_offset += word_wp_offset

        return {
            "input_ids": ft.as_folded_tensor(
                input_ids,
                data_dims=("context", "subword"),
                full_names=("context", "subword"),
                dtype=torch.long,
            ),
            "word_offsets": ft.as_folded_tensor(
                word_offsets,
                data_dims=("word",),
                full_names=("sample", "context", "word"),
                dtype=torch.long,
            ),
            "word_indices": torch.as_tensor(word_indices, dtype=torch.long),
            "empty_word_indices": torch.as_tensor(empty_word_indices, dtype=torch.long),
            "stats": {
                "tokens": sum(batch["stats"]["tokens"]),
                "words": sum(batch["stats"]["words"]),
                "contexts": sum(batch["stats"]["contexts"]),
            },
        }

    def forward(self, batch: TransformerBatchInput) -> TransformerBatchOutput:
        device = batch["input_ids"].device
        input_ids = batch["input_ids"].as_tensor()
        attention_mask = batch["input_ids"].mask
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        auto_batch_size = device.type == "cuda" and self.max_tokens_per_device == "auto"
        trial_idx = 1
        while True:
            total_tokens = input_ids.numel()
            if auto_batch_size:  # pragma: no cover
                max_tokens = INITIAL_MAX_TOKENS_PER_DEVICE
                torch.cuda.synchronize(device)
                total_mem = torch.cuda.get_device_properties(device).total_memory
                allocated_mem = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                free_mem = total_mem - allocated_mem

                if self._mem_per_unit is not None:
                    max_tokens = max(1, int(free_mem // self._mem_per_unit))
            else:
                max_tokens = (
                    self.max_tokens_per_device
                    if self.max_tokens_per_device != "auto"
                    else INITIAL_MAX_TOKENS_PER_DEVICE
                )

            max_windows = max(1, max_tokens // input_ids.size(1))
            total_windows = input_ids.size(0)
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
                    torch.cuda.empty_cache()
                    self._mem_per_unit = (free_mem / max_windows) * 1.5
                    trial_idx += 1
                    continue
                raise
            break

        # mask = batch["mask"].clone()
        # word_embeddings = torch.zeros(
        #     (mask.size(0), mask.size(1), wordpiece_embeddings.size(2)),
        #     dtype=torch.float,
        #     device=device,
        # )
        # embeddings_plus_empty = torch.cat(
        #     [
        #         wordpiece_embeddings.view(-1, wordpiece_embeddings.size(2)),
        #         self.empty_word_embedding,
        #     ],
        #     dim=0,
        # )
        word_embeddings = torch.nn.functional.embedding_bag(
            input=batch["word_indices"],
            weight=wordpiece_embeddings.reshape(-1, wordpiece_embeddings.size(2)),
            offsets=batch["word_offsets"],
        )
        word_embeddings[batch["empty_word_indices"]] = self.empty_word_embedding
        return {
            "embeddings": word_embeddings.refold("context", "word"),
        }

    @staticmethod
    def align_words_with_trf_tokens(doc, trf_char_indices):
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
            word_lengths[word_i] = length
        return word_tokens, word_lengths
