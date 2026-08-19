"""`eds.tnm` pipeline."""

from typing import Dict, Iterable, List, Optional, Tuple, Union

from pydantic import ValidationError
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipes.base import BaseNERComponent, SpanSetterArg
from edsnlp.utils.filter import filter_spans
from edsnlp.utils.typing import cast

from .model import TNM
from .patterns import default_banned_words, tnm_pattern


class TNMMatcher(BaseNERComponent):
    """
    The `eds.tnm` component extracts
    [TNM](https://enwp.org/wiki/TNM_staging_system) staging mentions from
    clinical documents and decomposes them into structured attributes.

    ## Extraction logic

    A span is extracted when **at least one** of the following conditions holds:

    - **T + N/M/R present**: the T component is followed by at least one of
      N (node), M (metastasis), or R (resection), with any delimiter
      (space, comma, slash, newline) between them.
    - **Standalone qualified T**: the T component carries both a prefix
      (e.g. `p`, `c`, `yp`) *and* a specification (e.g. `a`, `b`, `mi`),
      even without an N/M/R component.

    The pattern is **case-insensitive** (`pT2N1M0`, `pt2n1m0` and `PT2N1M0`
    are all matched). Delimiters between components can be spaces, commas,
    slashes, or newlines (e.g. `pT2 / N1 / M0`, `pT2,N1,M0`).

    ## Decomposition

    Each matched span is parsed into a `TNM` Pydantic model stored on
    `span._.tnm`. The following fields are extracted:

    | Field                      | Description                              |
    |----------------------------|------------------------------------------|
    | `tumour_prefix`            | Modifier prefix for T (c/p/y/r/a/u/m/s) |
    | `tumour`                   | T stage: 0–4, `is`, `x`                 |
    | `tumour_specification`     | T sub-spec: a/b/c/d/mi/x                |
    | `tumour_suffix`            | Parenthesised qualifier, e.g. `(m)`→`m` |
    | `node_prefix`              | Modifier prefix for N                    |
    | `node`                     | N stage: 0–4, `x`                       |
    | `node_specification`       | N sub-spec: mi/sn/i±/mol±/…             |
    | `node_suffix`              | Parenthesised qualifier for N            |
    | `metastasis_prefix`        | Modifier prefix for M                    |
    | `metastasis`               | M stage: 0–3, `x`                       |
    | `metastasis_specification` | Metastasis site: PUL/OSS/HEP/…          |
    | `metastasis_suffix`        | Parenthesised qualifier for M            |
    | `pleura`                   | PL stage 0–3 (lung cancer)               |
    | `resection_prefix`         | Modifier prefix for R                    |
    | `resection`                | Resection completeness: 0–2, `x`        |
    | `resection_specification`  | R sub-spec: is/cy+                       |
    | `resection_loc`            | Resection location qualifier             |
    | `resection_suffix`         | Parenthesised qualifier for R            |

    !!! note "Specification normalisation"
        Parenthesised specifications such as `(sn)` or `(mi)` are stored
        with their parentheses in the raw field but are stripped in `norm()`,
        so `N0(sn)` normalises to `N0sn`.

    ## Normalised form

    `span._.tnm.norm()` returns a compact canonical string that concatenates
    all non-`None` components, stripping delimiters and surrounding whitespace:

    ```
    {tumour_prefix}T{tumour}{tumour_specification}{tumour_suffix}
    {node_prefix}N{node}{node_specification}{node_suffix}
    {metastasis_prefix}M{metastasis}{metastasis_specification}
    PL{pleura}
    {resection_prefix}R{resection}{resection_specification}{resection_loc}
    ```

    This value is also stored on `span.kb_id_` for downstream filtering.

    Examples
    --------
    ```{ .python .no-check }
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(eds.tnm())

    text = "Conclusion : pT2c N1mi M0 R0"

    doc = nlp(text)
    doc.ents
    # Out: (pT2c N1mi M0 R0,)

    ent = doc.ents[0]
    ent._.tnm.norm()
    # Out: 'pT2cN1miM0R0'

    ent._.tnm.dict()
    # Out: {
    #   'tumour_prefix': 'p',
    #   'tumour': '2',
    #   'tumour_specification': 'c',
    #   'tumour_suffix': None,
    #   'node_prefix': None,
    #   'node': '1',
    #   'node_specification': 'mi',
    #   'node_suffix': None,
    #   'metastasis_prefix': None,
    #   'metastasis': '0',
    #   'metastasis_specification': None,
    #   'metastasis_suffix': None,
    #   'pleura': None,
    #   'resection_prefix': None,
    #   'resection': '0',
    #   'resection_specification': None,
    #   'resection_loc': None,
    #   'resection_suffix': None,
    # }
    ```

    Parameters
    ----------
    nlp : Optional[PipelineProtocol]
        The pipeline object
    name : str
        The name of the pipe
    pattern : Optional[Union[List[str], str]]
        The regex pattern used to match TNM spans. Defaults to
        `tnm_pattern`, which handles case-insensitive matching,
        multiple delimiter styles, and a logic filter that rejects
        false positives.
    banned_words : Optional[Iterable[str]]
        Lowercase, whitespace- and comma-free forms that must never be
        returned as TNM mentions. Defaults to `default_banned_words`.
        Pass an empty list to disable the post-filter.
    attr : str
        Attribute to match on, e.g. `TEXT`, `NORM`.
    label : str
        Label name used for the `Span` object and the `span._.<label>`
        extension.
    span_setter : SpanSetterArg
        How to set matches on the doc.

    Authors and citation
    --------------------
    The TNM pipe was originally developed by S. Priou, B. Rance and
    E. Kempf ([@kempf:hal-03519085]).
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Optional[PipelineProtocol],
        name: str = "tnm",
        *,
        pattern: Optional[Union[List[str], str]] = tnm_pattern,
        banned_words: Optional[Iterable[str]] = None,
        attr: str = "TEXT",
        label: str = "tnm",
        span_setter: SpanSetterArg = {"ents": True, "tnm": True},
    ):
        self.label = label

        super().__init__(nlp=nlp, name=name, span_setter=span_setter)

        if isinstance(pattern, str):
            pattern = [pattern]

        self.banned_words = frozenset(
            default_banned_words if banned_words is None else banned_words
        )

        self.regex_matcher = RegexMatcher(attr=attr, alignment_mode="expand")
        self.regex_matcher.add(self.label, pattern)

    def set_extensions(self) -> None:
        """
        Set spaCy extensions
        """
        super().set_extensions()

        if not Span.has_extension(self.label):
            Span.set_extension(self.label, default=None)

    def process(self, doc: Doc) -> List[Tuple[Span, Dict[str, str]]]:
        """
        Find TNM mentions in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        spans:
            list of (span, groupdict) tuples
        """

        spans = self.regex_matcher(
            doc,
            as_spans=True,
            return_groupdict=True,
        )

        filtered_spans = []
        for span, gd in spans:
            text = span.text
            clean = text.replace(" ", "").replace("\n", "").replace(",", "")
            if clean.strip().lower() in self.banned_words:
                continue
            if (
                # we keep it if it's longer than 2 chars
                len(clean) > 2
                # or shorter but there is no space, and it starts w/ a lowercase letter
                # to avoid cases like "a  T" or "PT"
                or (not text[1:2].isspace() and text[0:1].islower())
            ):
                filtered_spans.append((span, gd))

        # filter_spans only returns a tuple when return_discarded is set
        return filter_spans(filtered_spans)  # type: ignore[return-value]

    def parse(self, spans: List[Tuple[Span, Dict[str, str]]]) -> List[Span]:
        """
        Parse TNM mentions using the groupdict returned by the matcher.

        Parameters
        ----------
        spans : List[Tuple[Span, Dict[str, str]]]
            List of tuples containing the spans and groupdict
            returned by the matcher.

        Returns
        -------
        List[Span]
            List of processed spans, with `span._.<label>` and `span.kb_id_` set.
        """

        for span, groupdict in spans:
            try:
                value = cast(TNM, groupdict)
            except ValidationError:  # pragma: no cover
                value = cast(TNM, {})

            span._.set(self.label, value)
            span.kb_id_ = value.norm()

        return [span for span, _ in spans]

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags TNM mentions.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for TNM
        """
        spans = self.process(doc)
        spans = self.parse(spans)
        self.set_spans(doc, spans)
        return doc
