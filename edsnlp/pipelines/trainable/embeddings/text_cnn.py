from typing import Any, Dict, Optional, Sequence

import torch
from typing_extensions import TypedDict

from edsnlp import registry
from edsnlp.core.torch_component import BatchInput, TorchComponent
from edsnlp.utils.torch import ActivationFunction

from ..layers.text_cnn import NormalizationPlacement, TextCNN
from .typing import WordEmbeddingBatchOutput

TextCNNBatchInput = TypedDict(
    "TextCNNBatchInput",
    {
        "embeddings": torch.Tensor,
        "mask": torch.Tensor,
    },
)


@registry.factory.register("eds.text_cnn")
class TextCNNEncoder(TorchComponent[WordEmbeddingBatchOutput, BatchInput]):
    def __init__(
        self,
        nlp,
        name: str,
        embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput],
        output_size: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        activation: ActivationFunction = "relu",
        residual: bool = True,
        normalize: NormalizationPlacement = "pre",
    ):
        """
        Parameters
        ----------
        nlp: PipelineProtocol
            The spaCy Language object
        name: str
            The name of the component
        embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput]
            Embedding module to apply to the input
        output_size: Optional[int]
            Size of the output embeddings
            Defaults to the `input_size`
        out_channels: int
            Number of channels
        kernel_sizes: Sequence[int]
            Window size of each kernel
        activation: str
            Activation function to use
        residual: bool
            Whether to use residual connections
        normalize: NormalizationPlacement
            Whether to normalize before or after the residual connection
        """
        super().__init__(nlp, name)
        self.name = name
        self.embedding = embedding
        self.output_size = output_size or embedding.output_size
        self.module = TextCNN(
            input_size=self.embedding.output_size,
            output_size=self.output_size,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            activation=activation,
            residual=residual,
            normalize=normalize,
        )

    def preprocess(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return self.embedding.preprocess(doc)

    def collate(
        self, batch: Dict[str, Sequence[Any]], device: torch.device
    ) -> BatchInput:
        return self.embedding.collate(batch, device)

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
        embedding_results = self.embedding.module_forward(batch)
        convoluted = self.module(
            embedding_results["embeddings"],
            embedding_results["mask"],
        )
        return {
            "embeddings": convoluted,
            "mask": embedding_results["mask"],
        }


create_component = TextCNNEncoder
