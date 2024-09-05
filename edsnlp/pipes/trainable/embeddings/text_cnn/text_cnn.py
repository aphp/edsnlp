from typing import Optional, Sequence

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
        "embeddings": torch.Tensor,
        "mask": torch.Tensor,
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

    def forward(self, batch: BatchInput) -> WordEmbeddingBatchOutput:
        """
        Encode embeddings with a 1d convolutional network

        Parameters
        ----------
        batch: WordEmbeddingBatchOutput
            - embeddings: embeddings of shape (batch_size, seq_len, input_size)
            - mask: mask of shape (batch_size, seq_len)

        Returns
        -------
        WordEmbeddingBatchOutput
            - embeddings: encoded embeddings of shape (batch_size, seq_len, input_size)
            - mask: (same) mask of shape (batch_size, seq_len)
        """
        embedding = self.embedding(batch["embedding"])["embeddings"]
        embedding = embedding.refold("context", "word")
        convoluted = (
            self.module(
                embedding.as_tensor(),
                embedding.mask,
            )
            if embedding.size(0) > 0
            else embedding
        )
        return {
            "embeddings": embedding.with_data(convoluted),
        }
