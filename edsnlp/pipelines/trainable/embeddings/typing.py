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


class EmbeddingComponent(
    Generic[BatchInput], TorchComponent[WordEmbeddingBatchOutput, BatchInput]
):
    span_getter: Optional[SpanGetter]
