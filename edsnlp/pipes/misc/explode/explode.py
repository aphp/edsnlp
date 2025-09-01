from typing import Iterable, List, Optional, Tuple

from spacy.tokens import Doc, Span

import edsnlp
from edsnlp import Pipeline
from edsnlp.utils.span_getters import SpanGetterArg, get_spans_with_group


@edsnlp.registry.factory.register("eds.explode", spacy_compatible=False)
class Explode:
    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "explode",
        *,
        span_getter: SpanGetterArg = {"ents": True},
        filter_expr: Optional[str] = None,
    ):
        """
        Explode a Doc into multiple distinct Doc objects, one per span retrieved through
        the `span_getter` : each span becomes alone in its own Doc. Note that entities
        that are not selected by the `span_getter` will be lost in the new docs.

        !!! warning "Not for pipelines"

            This component is not meant to be used in a pipeline, but rather
            as a preprocessing step when dealing with a stream of documents
            as in the example below.

        !!! note "Difference with `eds.split`"

            While `eds.split` breaks a document into smaller chunks based on length or
            regex rules, `eds.explode` creates a separate document for each selected
            span. This means `eds.split` is typically used for segmenting text for
            context size or processing constraints, whereas `eds.explode` is designed
            for span-level tasks that require span-level mixing, like training span
            classifiers, ensuring that each span is isolated in its own document while
            preserving the original context.

        Examples
        --------
        ```python
        import edsnlp.pipes as eds
        from edsnlp.data.converters import MarkupToDocConverter

        converter = MarkupToDocConverter(
            preset="xml",
            # Put xml annotated spans in distinct doc.spans[label] groups
            span_setter={"*": True},
        )
        doc = converter(
            "Le <person>patient</person> a mal au <body_part>bras</body_part>, à la "
            "<body_part>jambe</body_part> et au <body_part>torse</body_part>"
        )

        exploder = eds.explode(span_getter=["body_part"])
        print(doc.text, "->", doc.spans)
        # Out: Le patient a mal au bras, à la jambe et au torse -> {'person': [patient], 'body_part': [bras, jambe, torse]}

        for new_doc in exploder(doc):
            print(new_doc.text, "->", new_doc.spans)
        # Out: Le patient a mal au bras, à la jambe et au torse -> {'person': [], 'body_part': [bras]}
        # Out: Le patient a mal au bras, à la jambe et au torse -> {'person': [], 'body_part': [jambe]}
        # Out: Le patient a mal au bras, à la jambe et au torse -> {'person': [], 'body_part': [torse]}
        ```

        Parameters
        ----------
        nlp: Optional[Pipeline]
            The pipeline object
        name: str
            Name of the pipe
        span_getter: SpanGetterArg
            The span getter to use to retrieve spans from the Doc.
            Default is `{"ents": True}` which retrieves all entities in `doc.ents`.
        filter_expr: Optional[str]
            An optional filter expression to filter the produced documents. The callable
            expects a single argument, the new Doc, and should return a boolean.
        """  # noqa: E501
        self.span_getter = span_getter
        self.filter_fn = eval(f"lambda doc:{filter_expr}") if filter_expr else None

    def __call__(self, doc: Doc) -> Iterable[Doc]:
        for new_doc in self.explode_doc(doc):
            if not self.filter_fn or self.filter_fn(new_doc):
                yield new_doc

    def explode_doc(self, doc: Doc) -> Iterable[Doc]:
        """
        Yield a sequence of docs, one per span returned by the getter.
        """
        base = doc.copy()
        span_pairs: List[Tuple[Span, str]] = list(
            get_spans_with_group(base, self.span_getter)
        )
        # Nothing to explode: return the original doc unchanged
        if not span_pairs:
            yield doc
            return

        for name in base.spans:
            base.spans[name] = []
        base.ents = []

        for span, group in span_pairs:
            if group == "ents":
                base.ents = [span]
                new_doc = base.copy()
                base.ents = []
            else:
                base.spans[group] = [span]
                new_doc = base.copy()
                base.spans[group] = []

            yield new_doc
