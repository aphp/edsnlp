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
        # Word embeddings use the largest subword position as a deterministic value
        if self.word_pooling_mode == "mean":
            inputs = ft.as_folded_tensor(
                inputs.refold("sample", "word", "token").as_tensor().max(-1).values,
                lengths=list(inputs.lengths)[:-1],
                data_dims=("sample", "word"),
                full_names=("sample", "context", "word"),
            )
        return {"inputs": inputs, "out_structure": inputs.lengths}

    def forward(self, batch):
        return {
            "embeddings": batch["inputs"]
            .unsqueeze(-1)
            .expand(-1, -1, self.output_size)
            .float(),
        }
