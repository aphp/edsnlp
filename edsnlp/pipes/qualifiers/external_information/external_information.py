import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import (
    BaseSpanAttributeClassifierComponent,
    SpanGetterArg,
)
from edsnlp.utils.bindings import make_binding_getter
from edsnlp.utils.span_getters import (
    get_spans,
)


@dataclass
class ExternalInformation:
    """
    Parameters
    ----------
    doc_attr: str
        The elements under this attribute should be
        a list of dicts with keys `value` and `class`
        (List[Dict[str, Any]]).

        ### Example:
        ```python
        import datetime

        doc_attr = "_.context_dates"
        context_dates = [
            {"value": datetime.datetime(2024, 2, 15), "class": "irm"},
            {"value": datetime.datetime(2024, 2, 7), "class": "biopsy"},
        ]
        ```

    span_attribute = "_.date.to_datetime()"

    threshold = datetime.timedelta(days=0)

    reduce: str = "all", the way to aggregate the matches
        one of ["all", "one_only", "closest"]

    comparison_type: str = "similarity", the way to compare the values.
    One of ["similarity", "exact_match"]
    """

    doc_attr: str
    span_attribute: str
    threshold: Union[float, dt.timedelta]
    reduce: str = "all"  # "one_only" , "closest" # TODO: implement
    comparison_type: str = "similarity"  # "exact_match"


class ExternalInformationQualifier(BaseSpanAttributeClassifierComponent):
    """
    The `eds.external_information_qualifier` pipeline component qualifies spans
    in a document based on external information and a defined distance to these
    contextual/external elements as in Distant Supervision (http://deepdive.stanford.edu/distant_supervision).

    Parameters
    ----------
    nlp : PipelineProtocol
        The spaCy pipeline object.
    name : Optional[str], default="distant_qualifier"
        The name of the component.
    span_getter : SpanGetterArg
        The function or callable to get spans from the document.
    external_information : Dict[str, ExternalInformation]
        A dictionary where keys are the names of the attributes to set on spans,
        and values are ExternalInformation objects defining the context and comparison
        settings.

        ??? note "`ExternalInformation`"
            ::: edsnlp.pipes.qualifiers.external_information.external_information.ExternalInformation
                options:
                    heading_level: 1
                    only_parameters: "no-header"
                    skip_parameters: []
                    show_source: false
                    show_toc: false

    Methods
    -------
    set_extensions()
        Sets custom extensions on the Span object for each context name.
    annotate(ent, name, value)
        Annotates a span with a given value.
    distance(spans, ctx_spans_values)
        Computes the distance between spans and context values.
    threshold(distances, threshold)
        Applies a threshold to the computed distances.
    mask_to_dict(idx_x, idx_y)
        Converts mask indices to a dictionary.
    reduce(mask, reduce_mode)
        Reduces the mask based on the specified mode.
    annotate(labels, filtered_spans, ctx_classes, name)
        Annotates spans with labels based on the reduced mask.
    __call__(doc: Doc) -> Doc
        Processes the document, qualifying spans based on their distance to context
        elements.
    """  # noqa: E501

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: Optional[str] = "external_information_qualifier",
        *,
        span_getter: SpanGetterArg,
        external_information: Dict[str, ExternalInformation],
    ):
        """
        Initializes the ExternalInformationQualifier component.

        Parameters
        ----------
        nlp : PipelineProtocol
            The spaCy pipeline object.
        name : Optional[str], default="external_information_qualifier"
            The name of the component.
        span_getter : SpanGetterArg
            The function or callable to get spans from the document.
        external_information : Dict[str, ExternalInformationQualifier]
            A dictionary where keys are the names of the attributes to set on spans,
            and values are ExternalInformationQualifier objects defining the context and
            comparison settings.
        """
        for key, context in external_information.items():
            if isinstance(context, dict):
                external_information[key] = ExternalInformation(**context)
        self.distant_context = external_information

        super().__init__(nlp, name, span_getter=span_getter)

    def set_extensions(self) -> None:
        """
        Sets custom extensions on the Span object for each context name.
        """
        for name in self.distant_context.keys():
            if not Span.has_extension(name):
                Span.set_extension(name, default=None)

    def numeric_distance(self, spans, ctx_spans_values):
        """
        Computes the distance between spans and context values.

        Parameters
        ----------
        spans : List
            The list of span attributes.
        ctx_spans_values : List
            The list of context values.

        Returns
        -------
        np.ndarray
            The computed distances.
        """
        doc_elements = np.array(spans)  # shape: N
        ctx_elements = np.array(ctx_spans_values)  # shape: M
        distances = doc_elements[:, None] - ctx_elements[None, :]  # shape: N x M

        return distances

    def exact_match(self, spans, ctx_spans_values):
        """
        Computes the exact match between spans and context values.

        Parameters
        ----------
        spans : List
            The list of span attributes.
        ctx_spans_values : List
            The list of context values.

        Returns
        -------
        np.ndarray
            A mask indicating which spans match the context values.
        """
        doc_elements = np.array(spans)  # shape: N
        ctx_elements = np.array(ctx_spans_values)  # shape: M
        distances = doc_elements[:, None] == ctx_elements[None, :]  # shape: N x M

        return distances

    def threshold(self, distances: np.ndarray, threshold: Union[float, dt.timedelta]):
        """
        Applies a threshold to the computed distances.

        Parameters
        ----------
        distances : np.ndarray
            The computed distances.
        threshold : Union[float, dt.timedelta]
            The threshold value.

        Returns
        -------
        np.ndarray
            A mask indicating which distances are within the threshold.
        """
        mask = np.abs(distances) <= threshold
        return mask

    def mask_to_dict(self, idx_x: np.ndarray, idx_y: np.ndarray):
        """
        Converts mask indices to a dictionary.

        Parameters
        ----------
        idx_x : np.ndarray
            The indices of the spans.
        idx_y : np.ndarray
            The indices of the context values.

        Returns
        -------
        Dict[int, List[int]]
            A dictionary mapping span indices to context value indices.
        """
        result = {}
        for x, y in zip(idx_x, idx_y):
            if x not in result:
                result[x] = []
            result[x].append(y)
        return result

    def reduce(self, mask: np.ndarray, reduce_mode: str):
        """
        Reduces the mask based on the specified mode.

        Parameters
        ----------
        mask : np.ndarray
            The mask indicating which distances are within the threshold.
        reduce_mode : str
            The mode to use for reducing the mask.
            One of ["all", "one_only", "closest"].

        Returns
        -------
        Dict[int, List[int]]
            A dictionary mapping span indices to context value indices.
        """
        if reduce_mode == "all":
            idx_x, idx_y = np.nonzero(mask)

            result = self.mask_to_dict(idx_x, idx_y)
            return result
        else:
            raise NotImplementedError

    def annotate(
        self,
        labels: Dict[int, List[int]],
        filtered_spans: List[Span],
        ctx_classes: List[Union[str, int]],
        name: str,
    ):
        """
        Annotates spans with labels based on the reduced mask.

        Parameters
        ----------
        labels : Dict[int, List[int]]
            A dictionary mapping span indices to context value indices.
        filtered_spans : List[Span]
            The list of filtered spans.
        ctx_classes : List[Union[str, int]]
            The list of context classes.
        name : str
            The name of the attribute to set.
        """
        for key, values in labels.items():
            span = filtered_spans[key]
            label_names = [ctx_classes[j] for j in values]
            span._.set(name, label_names)

    def __call__(self, doc: Doc) -> Doc:
        """
        Processes the document, qualifying spans based on their distance
        to context elements.

        Parameters
        ----------
        doc : Doc
            The spaCy document to process.

        Returns
        -------
        Doc
            The processed document with qualified spans.
        """
        for name, context in self.distant_context.items():
            # Get spans to qualify and their attributes
            doc_spans = list(get_spans(doc, self.span_getter))
            binding_getter_span_attr = make_binding_getter(context.span_attribute)

            filtered_spans = [
                span
                for span in doc_spans
                if not pd.isna(binding_getter_span_attr(span))
            ]

            filtered_spans_attr = [
                binding_getter_span_attr(span) for span in filtered_spans
            ]

            # Get context to annotate distantly
            binding_getter_doc_attr = make_binding_getter(context.doc_attr)
            context_doc: Optional[List[Dict[str, Any]]] = binding_getter_doc_attr(doc)
            if isinstance(context_doc, list):
                ctx_values = [i.get("value") for i in context_doc]  # values to look for
                ctx_classes = [i.get("class") for i in context_doc]  # classes to assign
                if len(ctx_values) > 0:
                    instance_type = type(ctx_values[0])
                    assert isinstance(
                        ctx_values[0], (dt.datetime, dt.date, str, float, int)
                    ), (
                        "Values should be (dt.datetime, dt.date, str, float, int)."
                        "Future: add support for"
                        " other types"
                    )

                    # Compute distance
                    if (
                        context.comparison_type == "similarity"
                        and instance_type is not str
                    ):
                        distances = self.numeric_distance(
                            filtered_spans_attr, ctx_values
                        )
                        mask = self.threshold(distances, context.threshold)
                        labels = self.reduce(mask, context.reduce)
                    elif (
                        context.comparison_type == "exact_match"
                        and instance_type is str
                    ):
                        mask = self.exact_match(filtered_spans_attr, ctx_values)
                        labels = self.reduce(mask, context.reduce)
                    else:
                        raise NotImplementedError

                    # Qualify / Annotate
                    self.annotate(labels, filtered_spans, ctx_classes, name)

        return doc
