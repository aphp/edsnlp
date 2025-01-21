import random
import re
from typing import Iterable, Optional

from spacy.tokens import Doc, Span

import edsnlp
from edsnlp import Pipeline

EMPTY = object()


def make_shifter(start, end, new_doc):
    cache = {}

    def rec(obj):
        if isinstance(obj, Span):
            if obj in cache:
                return cache[obj]
            if obj.end > start and obj.start < end:
                res = Span(
                    new_doc,
                    max(0, obj.start - start),
                    min(obj.end - start, end - start),
                    obj.label,
                )
            else:
                res = EMPTY
            cache[obj] = res
        elif isinstance(obj, (list, tuple, set)):
            res = type(obj)(
                filter(
                    lambda x: x is not EMPTY,
                    (rec(span) for span in obj),
                )
            )
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                new_v = rec(v)
                if new_v is not EMPTY:
                    res[k] = new_v
        else:
            res = obj
        return res

    return rec


def subset_doc(doc: Doc, start: int, end: int) -> Doc:
    """
    Subset a doc given a start and end index.

    Parameters
    ----------
    doc: Doc
        The doc to subset
    start: int
        The start index
    end: int
        The end index

    Returns
    -------
    Doc
    """
    new_doc = doc[start:end].as_doc()

    shifter = make_shifter(start, end, new_doc)

    char_beg = doc[start].idx if start < len(doc) else 0
    char_end = doc[end - 1].idx + len(doc[end - 1].text)
    for k, val in list(doc.user_data.items()):
        new_value = shifter(val)
        if k[0] == "._." and new_value is not EMPTY:
            new_doc.user_data[
                (
                    k[0],
                    k[1],
                    None if k[2] is None else max(0, k[2] - char_beg),
                    None if k[3] is None else min(k[3] - char_beg, char_end - char_beg),
                )
            ] = new_value

    for name, group in doc.spans.items():
        new_doc.spans[name] = shifter(list(group))

    return new_doc


@edsnlp.registry.factory.register("eds.split", spacy_compatible=False)
class Split:
    def __init__(
        self,
        nlp: Optional[Pipeline] = None,
        name: str = "split",
        *,
        max_length: int = 0,
        regex: Optional[str] = "\n{2,}",
        filter_expr: Optional[str] = None,
        randomize: float = 0.0,
    ):
        """
        The `eds.split` component splits a document into multiple documents
        based on a regex pattern or a maximum length.

        !!! warning "Not for pipelines"

            This component is not meant to be used in a pipeline, but rather
            as a preprocessing step when dealing with a stream of documents
            as in the example below.

        Examples
        --------

        ```python
        import edsnlp, edsnlp.pipes as eds

        # Create the stream
        stream = edsnlp.data.from_iterable(
            ["Sentence 1\\n\\nThis is another longer sentence more than 5 words"]
        )

        # Convert texts into docs
        stream = stream.map_pipeline(edsnlp.blank("eds"))

        # Apply the split component
        stream = stream.map(eds.split(max_length=5, regex="\\n{2,}"))

        print(" | ".join(doc.text.strip() for doc in stream))
        # Out:
        # Sentence 1 | This is another longer sentence | more than 5 words
        ```

        Parameters
        ----------
        nlp: Optional[Pipeline]
            The pipeline
        name: str
            The component name
        max_length: int
            The maximum length of the produced documents.
            If 0, the document will not be split based on length.
        regex: Optional[str]
            The regex pattern to split the document on
        filter_expr: Optional[str]
            An optional filter expression to filter the produced documents
        randomize: float
            The randomization factor to split the documents, to avoid
            producing documents that are all `max_length` tokens long
            (0 means all documents will have the maximum possible length
            while 1 will produce documents with a length varying between
            0 and `max_length` uniformly)
        """
        self.max_length = max_length
        self.regex = re.compile(regex) if regex else None
        self.filter_fn = eval(f"lambda doc:{filter_expr}") if filter_expr else None
        self.randomize = randomize

    def __call__(self, doc: Doc) -> Iterable[Doc]:
        for sub_doc in self.split_doc(doc):
            if sub_doc.text.strip():
                if not self.filter_fn or self.filter_fn(sub_doc):
                    yield sub_doc

    def split_doc(
        self,
        doc: Doc,
    ) -> Iterable[Doc]:
        """
        Split a doc into multiple docs of max_length tokens.

        Parameters
        ----------
        doc: Doc
            The doc to split

        Returns
        -------
        Iterable[Doc]
        """
        max_length = self.max_length

        if max_length <= 0 and self.regex is None:
            yield doc
            return

        start = 0
        end = 0
        # Split doc into segments between the regex matches
        matches = (
            [
                next(
                    m.end(g)
                    for g in range(self.regex.groups + 1)
                    if m.end(g) is not None
                )
                for m in self.regex.finditer(doc.text)
            ]
            if self.regex
            else []
        ) + [len(doc.text)]
        word_ends = doc.to_array("IDX") + doc.to_array("LENGTH")
        segments_end = word_ends.searchsorted([m for m in matches], side="right")

        for end in segments_end:
            # If the sentence adds too many tokens
            if end - start > max_length > 0:
                # But the current buffer too large
                while end - start > max_length:
                    subset_end = (
                        start + int(max_length * (random.random() ** self.randomize))
                        if self.randomize
                        else start + max_length
                    )
                    yield subset_doc(doc, start, subset_end)
                    start = subset_end
                yield subset_doc(doc, start, end)
                start = end

            if end > start:
                yield subset_doc(doc, start, end)
                start = end

        yield subset_doc(doc, start, end)
