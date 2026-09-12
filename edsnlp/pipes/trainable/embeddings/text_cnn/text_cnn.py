from typing import Any, Dict, Optional, Sequence

import foldedtensor as ft
import torch
from typing_extensions import Literal, TypedDict

from edsnlp.core.pipeline import Pipeline
from edsnlp.core.torch_component import BatchInput
from edsnlp.pipes.trainable.embeddings.typing import (
    WordContextualizerComponent,
    WordEmbeddingBatchOutput,
    WordEmbeddingComponent,
)
from edsnlp.pipes.trainable.layers.text_cnn import TextCnn
from edsnlp.utils.torch import ActivationFunction

TextCnnBatchInput = TypedDict(
    "TextCnnBatchInput",
    {
        "embedding": BatchInput,
        "word_indices": torch.LongTensor,
        "padded_length": int,
        "out_structure": ft.FoldedTensorLayout,
    },
)


class TextCnnEncoder(WordContextualizerComponent):
    """
    The `eds.text_cnn` component is a simple 1D convolutional network to contextualize
    word embeddings (as computed by the `embedding` component passed as argument).

    To be memory efficient when handling batches of variable-length sequences, this
    module employs sequence packing, while taking care of avoiding contamination between
    the different docs.

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : str
        The name of the component
    embedding : TorchComponent[WordEmbeddingBatchOutput, BatchInput]
        Embedding module to apply to the input
    output_size : Optional[int]
        Size of the output embeddings
        Defaults to the `input_size`
    out_channels : int
        Number of channels
    kernel_sizes : Sequence[int]
        Window size of each kernel
    activation : str
        Activation function to use
    residual : bool
        Whether to use residual connections
    normalize : Literal["pre", "post", "none"]
        Whether to normalize before or after the residual connection
    """

    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "text_cnn",
        *,
        embedding: WordEmbeddingComponent,
        output_size: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        activation: ActivationFunction = "relu",
        residual: bool = True,
        normalize: Literal["pre", "post", "none"] = "pre",
    ):
        sub_span_getter = getattr(embedding, "span_getter", None)
        if sub_span_getter is not None:  # pragma: no cover
            self.span_getter = sub_span_getter
        sub_context_getter = getattr(embedding, "context_getter", None)
        if sub_context_getter is not None:  # pragma: no cover
            self.context_getter = sub_context_getter

        super().__init__(nlp, name)
        self.embedding = embedding
        self.output_size = output_size or embedding.output_size
        self.module = TextCnn(
            input_size=self.embedding.output_size,
            output_size=self.output_size,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            activation=activation,
            residual=residual,
            normalize=normalize,
        )

    def collate(self, batch: Dict[str, Any]) -> TextCnnBatchInput:
        """
        Map words into a flat sequence with zero padding around each context
        """
        emb = self.embedding.collate(batch["embedding"])
        lengths = emb["out_structure"]["word"]
        max_kernel = max(conv.kernel_size[0] for conv in self.module.convolutions)
        padded_lengths = [length + max_kernel - 1 for length in lengths]
        layout = ft.FoldedTensorLayout(
            [[len(lengths)], padded_lengths],
            full_names=("context", "word"),
            data_dims=("word",),
        )
        contexts = torch.arange(len(lengths))
        word_indices, _, _ = layout.make_indices_ranges(
            begins=(contexts, max_kernel // 2),
            ends=(contexts, [length + max_kernel // 2 for length in lengths]),
            indice_dims=("context", "word"),
        )
        return {
            "embedding": emb,
            "word_indices": word_indices,
            "padded_length": sum(padded_lengths),
            "out_structure": ft.FoldedTensorLayout(
                emb["out_structure"],
                full_names=emb["out_structure"].full_names,
                data_dims=("word",),
            ),
        }

    def forward(self, batch: TextCnnBatchInput) -> WordEmbeddingBatchOutput:
        """
        Return flat contextualized words preserving virtual dimensions
        """
        embedding = self.embedding(batch["embedding"])["embeddings"].refold("word")
        return {
            "embeddings": embedding.with_data(
                self.module(
                    embedding.as_tensor(), batch["word_indices"], batch["padded_length"]
                )
            ),
        }
