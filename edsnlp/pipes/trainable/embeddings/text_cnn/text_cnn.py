from typing import Optional, Sequence

import torch
from typing_extensions import TypedDict

from edsnlp.core.torch_component import BatchInput
from edsnlp.pipes.trainable.embeddings.typing import (
    WordEmbeddingBatchOutput,
    WordEmbeddingComponent,
)
from edsnlp.pipes.trainable.layers.text_cnn import NormalizationPlacement, TextCnn
from edsnlp.utils.torch import ActivationFunction

TextCnnBatchInput = TypedDict(
    "TextCnnBatchInput",
    {
        "embeddings": torch.Tensor,
        "mask": torch.Tensor,
    },
)


class TextCnnEncoder(WordEmbeddingComponent):
    """
    The `eds.text_cnn` component is a simple 1D convolutional network to contextualize
    word embeddings (as computed by the `embedding` component passed as argument).

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
    normalize : NormalizationPlacement
        Whether to normalize before or after the residual connection
    """

    def __init__(
        self,
        nlp,
        name: str,
        embedding: WordEmbeddingComponent,
        output_size: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        activation: ActivationFunction = "relu",
        residual: bool = True,
        normalize: NormalizationPlacement = "pre",
    ):
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

    @property
    def span_getter(self):
        return self.embedding.span_getter

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
        embedding_results = self.embedding.module_forward(batch["embedding"])
        if embedding_results["embeddings"].size(0) == 0:
            return embedding_results

        convoluted = self.module(
            embedding_results["embeddings"],
            embedding_results["mask"],
        )
        return {
            "embeddings": convoluted,
            "mask": embedding_results["mask"],
        }
