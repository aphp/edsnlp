from typing import List, Optional

import pytest

pytest.importorskip("torch")

import foldedtensor as ft
import torch
from typing_extensions import Literal

from edsnlp import Pipeline
from edsnlp.pipes.trainable.embeddings.typing import WordEmbeddingComponent


class DummyEmbeddings(WordEmbeddingComponent[dict]):
    """
    For each word, embedding = (word idx in sent) * [1, 1, ..., 1] (size = dim)
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "fixed_embeddings",
        word_pooling_mode: Literal["mean", False] = "mean",
        *,
        dim: int,
    ):
        super().__init__(nlp, name)
        self.output_size = int(dim)
        self.word_pooling_mode = word_pooling_mode

    def preprocess(self, doc, *, contexts=None, prompts=()):
        if contexts is None:
            contexts = [doc[:]]

        inputs: List[List[List[int]]] = []
        total = 0

        for ctx in contexts:
            words = []
            for word in ctx:
                subwords = []
                for subword in word.text[::4]:
                    subwords.append(total)
                    total += 1
                words.append(subwords)
            inputs.append(words)

        return {
            "inputs": inputs,  # List[Context][Word] -> int
        }

    def collate(self, batch):
        # Flatten indices and keep per-(sample,context) lengths to refold later
        inputs = ft.as_folded_tensor(
            batch["inputs"],
            data_dims=("sample", "token"),
            full_names=("sample", "context", "word", "token"),
            dtype=torch.long,
        )
        item_indices = span_offsets = span_indices = None
        if self.word_pooling_mode == "mean":
            samples = torch.arange(max(inputs.lengths["sample"]))
            words = torch.arange(max(inputs.lengths["word"]))
            n_words = len(words)
            n_samples = len(samples)
            words = words[None, :].expand(n_samples, -1)
            samples = samples[:, None].expand(-1, n_words)
            words = words.masked_fill(
                ~inputs.refold("sample", "word", "token").mask.any(-1), 0
            )
            item_indices, span_offsets, span_indices = (
                inputs.lengths.make_indices_ranges(
                    begins=(samples, words),
                    ends=(samples, words + 1),
                    indice_dims=(
                        "sample",
                        "word",
                    ),
                    return_tensors="pt",
                )
            )
            span_offsets = ft.as_folded_tensor(
                span_offsets,
                data_dims=(
                    "sample",
                    "word",
                ),
                full_names=("sample", "context", "word"),
                lengths=list(inputs.lengths)[0:-1],
            )

        return {
            "out_structure": span_offsets.lengths
            if self.word_pooling_mode == "mean"
            else inputs.lengths,
            "inputs": inputs,
            "item_indices": item_indices,
            "span_offsets": span_offsets,
            "span_indices": span_indices,
        }

    def forward(self, batch):
        embeddings = (
            batch["inputs"]
            .unsqueeze(-1)
            .expand(-1, -1, self.output_size)
            .to(torch.float32)
        )
        print("shape before pool", embeddings.shape)
        if self.word_pooling_mode == "mean":
            embeddings = torch.nn.functional.embedding_bag(
                embeddings.view(-1, self.output_size),
                batch["item_indices"],
                offsets=batch["span_offsets"].view(-1),
                mode="max",
            )
            embeddings = batch["span_offsets"].with_data(
                embeddings.view(*batch["span_offsets"].shape, self.output_size)
            )
        return {
            "embeddings": embeddings,
        }
