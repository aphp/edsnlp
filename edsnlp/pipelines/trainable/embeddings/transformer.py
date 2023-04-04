import math
from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing_extensions import TypedDict

from edsnlp import registry
from edsnlp.core.torch_component import TorchComponent
from edsnlp.utils.torch import pad_2d

from .typing import WordEmbeddingBatchOutput

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


@registry.factory.register("eds.transformer")
class Transformer(TorchComponent[WordEmbeddingBatchOutput, TransformerBatchInput]):
    def __init__(
        self,
        nlp,
        name: str,
        model: Union[str, Path],
        window: int = 20,
        stride: int = 10,
    ):
        super().__init__(nlp, name)
        self.name = name
        self.transformer = AutoModelForMaskedLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.window = window
        self.stride = stride
        self.output_size = self.transformer.config.hidden_size
        self.empty_word_embedding = torch.nn.Parameter(
            torch.randn(
                (1, self.output_size),
            )
        )

    def preprocess(self, doc):
        preps = self.tokenizer(
            doc.text,
            is_split_into_words=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )

        word_tokens, word_lengths = self.align_words_with_trf_tokens(
            doc, preps.pop("offset_mapping")[1:-1]
        )

        bos, *input_ids, eos = preps["input_ids"]
        windows = [
            [
                bos,
                *input_ids[
                    window_i * self.stride : window_i * self.stride + self.window
                ],
                eos,
            ]
            for window_i in range(
                0, 1 + max(0, math.ceil((len(input_ids) - self.window) / self.stride))
            )
        ]

        return {
            "word_tokens": word_tokens,
            "word_lengths": word_lengths,
            "input_ids": windows,
        }

    def collate(self, preps, device: torch.device):
        input_ids = preps["input_ids"]
        token_window_indices = []
        words_offsets = []
        window_i = 0
        all_windows = []
        mask = []
        offset = 0
        window_max_size = max(len(w) for windows in preps["input_ids"] for w in windows)
        window_count = sum(len(windows) for windows in preps["input_ids"])
        for windows, doc_words_tokens, doc_words_lengths in zip(
            preps["input_ids"],
            preps["word_tokens"],
            preps["word_lengths"],
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

        token_window_indices = torch.as_tensor(
            token_window_indices, dtype=torch.long, device=device
        )

        input_ids = pad_2d(
            all_windows,
            pad=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).to(device),
            "token_window_indices": token_window_indices,
            "words_offsets": torch.as_tensor(
                words_offsets, dtype=torch.long, device=device
            ),
            "mask": pad_2d(mask, pad=False, dtype=torch.bool, device=device),
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

        for word_i, word in enumerate(doc):
            length = 0
            word_begin = word.idx
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


create_component = Transformer
