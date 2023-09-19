import math
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModel, AutoTokenizer
from typing_extensions import TypedDict

from edsnlp import Pipeline
from edsnlp.pipelines.trainable.embeddings.typing import EmbeddingComponent
from edsnlp.utils.span_getters import SpanGetterArg, get_spans
from edsnlp.utils.torch import pad_2d

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


class Transformer(EmbeddingComponent[TransformerBatchInput]):
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
    values are 510 and 255 respectively, which means that the model will process windows
    of 510 tokens, each separated by 255 tokens. Whenever a token appears in multiple
    windows, the embedding of the "most contextualized" occurrence is used, i.e. the
    occurrence that is the closest to the center of its window.

    Here is an overview how this works in a classifier model :

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
        before feeding them to the Transformer model (default: 510 = 512 - 2)
    stride: int
        The stride (distance between windows) to use when splitting long documents into
        smaller windows: (default: 510 / 2 = 255)
    max_tokens_per_device: int
        The maximum number of tokens that can be processed by the model on a single
        device. This does not affect the results but can be used to reduce the memory
        usage of the model, at the cost of a longer processing time.
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
        max_tokens_per_device: int = 128 * 128,
        span_getter: Optional[SpanGetterArg] = None,
    ):
        super().__init__(nlp, name)
        self.name = name
        self.transformer = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.window = window
        self.stride = stride
        self.output_size = self.transformer.config.hidden_size
        self.empty_word_embedding = torch.nn.Parameter(
            torch.randn(
                (1, self.output_size),
            )
        )
        self.max_tokens_per_device = max_tokens_per_device
        self.span_getter = span_getter

    def preprocess(self, doc):
        word_tokens = []
        word_lengths = []
        windows = []
        for span in get_spans(doc, self.span_getter):
            print("SPAN", span)
            preps = self.tokenizer(
                span.text,
                is_split_into_words=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )

            span_word_tokens, span_word_lengths = self.align_words_with_trf_tokens(
                span, preps.pop("offset_mapping")[1:-1]
            )
            word_tokens.append(span_word_tokens)
            word_lengths.append(span_word_lengths)

            bos, *input_ids, eos = preps["input_ids"]
            span_windows = [
                [
                    bos,
                    *input_ids[
                        window_i * self.stride : window_i * self.stride + self.window
                    ],
                    eos,
                ]
                for window_i in range(
                    0,
                    1 + max(0, math.ceil((len(input_ids) - self.window) / self.stride)),
                )
            ]
            windows.append(span_windows)

        return {
            "word_tokens": word_tokens,
            "word_lengths": word_lengths,
            "input_ids": windows,
        }

    def collate(self, preps):
        input_ids = [window for span in preps["input_ids"] for window in span]
        token_window_indices = []
        words_offsets = []
        window_i = 0
        all_windows = []
        mask = []
        offset = 0
        window_max_size = max(len(w) for windows in input_ids for w in windows)
        window_count = sum(len(windows) for windows in input_ids)
        for windows, doc_words_tokens, doc_words_lengths in zip(
            input_ids,
            chain.from_iterable(preps["word_tokens"]),
            chain.from_iterable(preps["word_lengths"]),
        ):
            all_windows.extend(windows)

            cols = list(
                range(
                    1,
                    min(1 + ((self.window - self.stride) // 2), len(windows[0]) - 1),
                )
            )
            rows = [window_i] * len(cols)
            window_i -= 1
            for window in windows:
                window_i += 1
                col_start = 1 + (self.window - self.stride) // 2
                last_cols = range(
                    col_start,
                    min(col_start + self.stride, len(window) - 1),
                )
                rows.extend([window_i] * len(last_cols))
                cols.extend(last_cols)

            last_cols = range(
                len(window) - ((len(window) - 2 - self.stride + 2) // 2),
                min(self.window + 1, len(window)),
            )
            cols.extend(last_cols)
            rows.extend([window_i] * len(last_cols))
            window_i += 1

            token_window_indices.extend(
                [
                    cols[i] + rows[i] * window_max_size
                    if i >= 0
                    else window_max_size * window_count
                    for i in doc_words_tokens
                ]
            )
            for length in doc_words_lengths:
                words_offsets.append(offset)
                offset += length
            mask.append([True] * len(doc_words_lengths))

        token_window_indices = torch.as_tensor(token_window_indices, dtype=torch.long)

        pad_id = self.tokenizer.pad_token_id
        input_ids = pad_2d(all_windows, pad=pad_id, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != pad_id,
            "token_window_indices": token_window_indices,
            "words_offsets": torch.as_tensor(words_offsets, dtype=torch.long),
            "mask": pad_2d(mask, pad=False, dtype=torch.bool),
        }

    def forward(self, batch):
        device = batch["input_ids"].device

        trf_result = self.transformer.base_model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        wordpiece_embeddings = trf_result.last_hidden_state

        mask = batch["mask"]
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
            input=batch["token_window_indices"],
            weight=embeddings_plus_empty,
            offsets=batch["words_offsets"],
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
        doc_start = doc[0].idx

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
