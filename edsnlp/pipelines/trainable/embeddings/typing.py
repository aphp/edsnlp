from typing import Generic, Optional

import torch
from typing_extensions import TypedDict

from edsnlp.core.torch_component import BatchInput, TorchComponent  # noqa: F401
from edsnlp.utils.span_getters import SpanGetter

WordEmbeddingBatchOutput = TypedDict(
    "WordEmbeddingBatchOutput",
    {
        "embeddings": torch.Tensor,
        "mask": torch.Tensor,
    },
)


class WordEmbeddingComponent(
    Generic[BatchInput], TorchComponent[WordEmbeddingBatchOutput, BatchInput]
):
    span_getter: Optional[SpanGetter]
    output_size: int


SpanEmbeddingBatchOutput = TypedDict(
    "SpanEmbeddingBatchOutput",
    {
        "embeddings": torch.Tensor,
        "mask": torch.Tensor,
    },
)


class SpanEmbeddingComponent(
    Generic[BatchInput], TorchComponent[SpanEmbeddingBatchOutput, BatchInput]
):
    span_getter: Optional[SpanGetter]
    output_size: int
