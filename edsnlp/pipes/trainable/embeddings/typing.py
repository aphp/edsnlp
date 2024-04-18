from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional

from foldedtensor import FoldedTensor
from spacy.tokens import Doc, Span
from typing_extensions import TypedDict

from edsnlp.core.torch_component import BatchInput, TorchComponent  # noqa: F401
from edsnlp.utils.span_getters import SpanGetter

WordEmbeddingBatchOutput = TypedDict(
    "WordEmbeddingBatchOutput",
    {
        "embeddings": FoldedTensor,
    },
)


class WordEmbeddingComponent(
    TorchComponent[WordEmbeddingBatchOutput, BatchInput],
    Generic[BatchInput],
):
    span_getter: Optional[SpanGetter]
    output_size: int

    if TYPE_CHECKING:

        def preprocess(
            self,
            doc: Doc,
            *,
            contexts: Optional[List[Span]],
            **kwargs,
        ) -> Dict[str, Any]:
            ...


class WordContextualizerComponent(
    Generic[BatchInput], WordEmbeddingComponent[BatchInput]
):
    span_getter: Optional[SpanGetter]
    output_size: int
    embedding: WordEmbeddingComponent


SpanEmbeddingBatchOutput = TypedDict(
    "SpanEmbeddingBatchOutput",
    {
        "embeddings": FoldedTensor,
    },
)


class SpanEmbeddingComponent(
    TorchComponent[SpanEmbeddingBatchOutput, BatchInput],
    Generic[BatchInput],
):
    span_getter: Optional[SpanGetter]
    output_size: int

    if TYPE_CHECKING:

        def preprocess(
            self,
            doc: Doc,
            *,
            spans: Optional[List[Span]],
            contexts: Optional[List[Span]],
            pre_aligned: bool = False,
            **kwargs,
        ) -> Dict[str, Any]:
            ...
