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

    Matching happens in two stages: a regex whose leading lookahead
    (`logic_filter`) decides *whether* a candidate is a TNM mention, then a
    post-filter that drops known lookalike abbreviations.

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

    ### What is matched

    | Input                    | Why                                     |
    |--------------------------|-----------------------------------------|
    | `pT2N1M0`                | T with N and M                          |
    | `T2N1`                   | T with N, prefix not required           |
    | `pT2 / N1 / M0`          | slash, comma and newline delimiters     |
    | `p Tx N1M 0`             | spaces inside and between components    |
    | `pT2b`                   | standalone T, prefix **and** spec       |
    | `pT1(m)N1M0`             | parenthesised T suffix (multifocal)     |
    | `pT1bN0(sn)`             | parenthesised N specification           |
    | `pT2N1(3/12)M0`          | examined/positive node ratio            |
    | `pT2N1M1PUL`             | metastasis site specification           |
    | `pT4N2R1(foie)`          | resection status with location          |
    | `pT2N1M0 PL1`            | pleural invasion (lung staging)         |
    | `pT2N1M0 (UICC 2017)`    | trailing classification version         |

    ### What is deliberately not matched

    | Input             | Why                                             |
    |-------------------|-------------------------------------------------|
    | `T2`, `pT2`       | bare T without spec and without N/M/R           |
    | `T2a`             | spec but no prefix and no N/M/R                 |
    | `PT`              | no T stage value                                |
    | `N1M0`, `pN1`     | the T component is mandatory                    |
    | `MTX`, `CTX`, `RTX`, `cyto`, `auto`, `atom` | in `banned_words` |

    The `banned_words` post-filter also drops any match of two characters or
    less that does not start with a lowercase letter, which removes the many
    `T`/`PT` fragments produced by uppercase headings and tables.

    !!! warning "Known limitations"
        - `a` (non-invasive papillary carcinoma) is not a recognised T stage
          value, alongside `0`-`4`, `is` and `x`, so a full mention such as
          `pTaN0M0` is missed. The standalone `pTa` would be rejected by the
          anchor rule anyway, like `pT2` or `pTis`. Adding `a` to the stage
          values is deliberately left out for now: accepting the letter `o`
          as a `0` already accounts for most of the observed false positives,
          and a bare letter as a stage value is expected to behave the same.
        - The anchor rule is a lookahead, so a component only has to
          *look* like an N/M/R to open it. `o` being a valid stage value, a
          following word starting with `no`, `mo` or `ro` opens it and the
          span degrades to the T alone: `pT2 nodulaire` yields `pT2`.
          `banned_words` catches the bare forms, not the prefixed ones. A
          follow-up will enforce the rule on the parsed components instead,
          which removes this class of false positives.
        - A component glued to another indicator breaks the trailing word
          boundary: `pT3(4)N2M0R0G1` yields the truncated span `pT3`, and
          `ypT1cN0R0M0TRG2` yields nothing.
        - Intercurrent noise, unusual spacing or non-standard prefixes break
          the component chain: `p T2 (40 mm) 9N+/20` and `iT3a iN0 Mx` (`i`,
          for incidental or imaging, is not a recognised prefix) yield
          nothing, and `T1c N0- M0` yields the truncated span `T1c`.

    ## Decomposition

    Each matched span is parsed into a `TNM` Pydantic model stored on
    `span._.tnm`. The following fields are extracted:

    | Field                      | Description                              |
    |----------------------------|------------------------------------------|
    | `tumour_prefix`            | Prefix for T: one or two of c/p/y/r/a/u/m/s |
    | `tumour`                   | T stage: 0–4, `is`, `x`                 |
    | `tumour_specification`     | T sub-spec: a/b/c/d/m/mi/x              |
    | `tumour_suffix`            | Parenthesised qualifier, e.g. `(m)`→`m` |
    | `node_prefix`              | Modifier prefix for N                    |
    | `node`                     | N stage: 0–4, `x`, `+`                  |
    | `node_specification`       | N sub-spec: mi/sn/i±/mol±/(3/12)/…      |
    | `node_suffix`              | Parenthesised qualifier for N            |
    | `metastasis_prefix`        | Modifier prefix for M                    |
    | `metastasis`               | M stage: 0–3, `x`, `+`                  |
    | `metastasis_specification` | Site (PUL/OSS/HEP/…) or marker (i+/mol+/cy+) |
    | `metastasis_suffix`        | Parenthesised qualifier for M            |
    | `pleura`                   | PL stage 0–3 or `x` (lung cancer)        |
    | `resection_prefix`         | Modifier prefix for R                    |
    | `resection`                | Resection completeness: 0–2, `x`, `+`   |
    | `resection_specification`  | R sub-spec: is/cy+                       |
    | `resection_loc`            | Resection location qualifier             |
    | `resection_suffix`         | Parenthesised qualifier for R            |
    | `version`                  | Classification: UICC/AJCC/ACCJ/TNM       |
    | `version_year`             | Classification year, expanded to 4 digits |

    Each component carries its **own** prefix: in `pT1 cN1 M0` the tumour is
    pathological while the node is clinical, and both are kept.

    !!! note "Specification normalisation"
        Parenthesised specifications such as `(sn)` or `(mi)` are stored
        with their parentheses in the raw field but are stripped in `norm()`,
        so `N0(sn)` normalises to `N0sn`.

    !!! note "Suffixes in the normalised form"
        The `_suffix` groups are permissive on purpose, so they also pick up
        free text: `pT2N1M0R0(marge saine)` stores `marge saine` in
        `resection_suffix`. Only suffixes that read as a TNM qualifier (one to
        three letters, e.g. the `(m)` of `pT1(m)`) are carried into `norm()`;
        anything else stays available on the field but is left out of the
        canonical string, so `span.kb_id_` remains usable for grouping.

    !!! note "The letter `o`"
        `o` and `O` are normalised to the digit `0`, but **only** in the
        numeric stage fields (`tumour`, `node`, `metastasis`, `pleura`,
        `resection`). Free-text fields keep their letters, so `N1(mol+)`
        stays `mol+` and `M1OSS` stays `OSS`.

    ## Normalised form

    `span._.tnm.norm()` returns a compact canonical string that concatenates
    all non-`None` components, stripping delimiters and surrounding whitespace:

    ```
    {tumour_prefix}T{tumour}{tumour_specification}{tumour_suffix}
    {node_prefix}N{node}{node_specification}{node_suffix}
    {metastasis_prefix}M{metastasis}{metastasis_specification}{metastasis_suffix}
    PL{pleura}
    {resection_prefix}R{resection}{resection_specification}{resection_loc}{resection_suffix}
    ({VERSION} {version_year})
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
    #   'version': None,
    #   'version_year': None,
    # }
    ```


    ## Migrating from the previous version

    The regex and the `TNM` model were rewritten. Extracted spans and `norm()`
    values are broadly compatible, and the two renamed fields keep a
    deprecated alias, so most code reading `span._.tnm.<field>` keeps
    working.

    ### Renamed fields

    | Before                   | Now             | Note                       |
    |--------------------------|-----------------|----------------------------|
    | `prefix`                 | `tumour_prefix` | Each component has its own |
    | `resection_completeness` | `resection`     | Was an `int`, now a `str`  |

    The old names still read, with a `DeprecationWarning`, so existing code
    keeps working. `resection_completeness` still returns an `int` for a
    numeric status, and a `str` for the `x` and `+` values the previous model
    could not represent.

    ```{ .python .no-check }
    tnm.prefix                 # -> deprecated, use tnm.tumour_prefix
    tnm.resection_completeness # -> deprecated, use tnm.resection
    ```

    ### Enums replaced by strings

    `Prefix`, `Tumour`, `Specification`, `Node`, `Metastasis` and `TnmEnum`
    were removed from `edsnlp.pipes.ner.tnm.model`. Every field now holds the
    raw matched text as a `str` (except `version_year`, an `int`).

    ```{ .python .no-check }
    # Before -- fields were enum members
    from edsnlp.pipes.ner.tnm.model import Tumour
    tnm.tumour is Tumour.score_2

    # Now -- fields are plain strings
    tnm.tumour == "2"
    ```

    The value space is not open, though. The pattern has always been the gate
    -- the previous one restricted stages just as narrowly (`[0-4o]|is` for T,
    `[0-3o]|x` for N, `[01o]|x` for M) and the enums merely duplicated that
    check in the model. Only the duplicate is gone, and the accepted values
    are those listed in the decomposition table above. What the enums could
    not represent -- `N4`, `M2`, `M3`, `Rx`, `R+`, metastasis site codes such
    as `PUL` -- is accepted now. Only the free-text fields are genuinely open:
    `*_suffix`, `resection_loc`, and node ratios such as `(3/12)`.

    ### New fields

    `node_prefix`, `metastasis_prefix`, `resection_prefix`,
    `metastasis_specification`, `metastasis_suffix`, `resection_specification`,
    `resection_loc`, `resection_suffix` and `pleura`. They default to `None`,
    so existing code keeps working -- but `norm()` and `span.kb_id_` now
    include them, which means a normalised value may be longer than before for
    the same text.

    ### Behaviour changes to be aware of

    - **Matching is case-insensitive.** `pt2n1m0` and `PT2N1M0` are now
      extracted; previously only certain case combinations were.
    - **A lone T is no longer extracted** unless it carries both a prefix and
      a specification (`pT2b`), or is followed by an N, M or R component.
      `pT2` and `pTx` used to match and no longer do. This is the main driver
      of the precision gain.
    - **`o` to `0` coercion is now restricted** to the numeric stage fields,
      so `M1OSS` keeps its `O`. Previously every field was coerced.
    - **A component prefix binds to its own component.** In `pT1 cN1 M0` the
      `c` is now `node_prefix`; it used to land in `tumour_specification`.
    - **`norm()` no longer concatenates free-text suffixes verbatim** --
      `pT1(grade 2)N1M0` used to give `pT1grade 2N1M0`. Only suffixes that
      read as a TNM qualifier are kept; the full text stays on the field.
    - **`banned_words`** is a new parameter. Pass an empty list to restore the
      unfiltered regex output.

    ## Evaluation

    The pipe was qualified before production use, on an initial sample of 20
    million clinical notes stratified by year, restricted to the ~5 million
    documents belonging to patients followed for cancer. Both samples were
    annotated by two physicians. Sampling used Neyman allocation over strata,
    with a minimum of 5 documents per stratum; the figures below are the
    corresponding stratum-weighted estimates.

    | Metric                  | Estimate | Interval        | Unit     | N   |
    |-------------------------|----------|-----------------|----------|-----|
    | Precision               | 98.64 %  | +/- 1 % (95 %)  | mention  | 366 |
    | Recall (entity level)   | 79.40 %  | +/- 1 % (99 %)  | mention  | 120 |
    | Recall (document level) | 95.53 %  | +/- 1 % (99 %)  | document | 120 |

    Document-level recall is the share of documents containing at least one
    TNM mention for which at least one mention is retrieved. It is much higher
    than the entity-level figure because staging is usually repeated within a
    report.

    Strata were built on the document type (pathology report / tumour board
    report vs. other), the oncological density of the issuing care unit, and
    -- for precision -- whether the mention carried only a T component, the
    configuration most prone to false positives. Recall strata additionally
    split on whether the pipe found a mention and whether the raw text
    contained the word `TNM`.

    **Precision errors.** 21 false positives out of 366 annotations. Most come
    from the rule accepting the letter `o` as a substitute for the digit `0`
    (`p t o m`, `TOM`, `TOC`, `CS tox`). The rest are interfering acronyms
    (`CMT1A`, `RT 3D`), numbering or temporal wording (`Tour 1`, `au T1`), and
    mentions whose format is a valid TNM but whose context is not.

    **Recall errors.** 27 false negatives, matching the patterns listed under
    Known limitations above. The isolated `M+` mention was excluded from the
    count after review, as it is not considered a valid TNM here; keeping it
    would lower the estimates to 57.60 % and 60.74 % respectively, since it
    falls in a stratum representing a third of the population.

    !!! note "Scope of these figures"
        The review was run on the first version of this pattern. Additional
        prefixes and specifications were added afterwards without re-running
        it, so the estimates are conservative.

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
    The `eds.tnm` component was originally developed by S. Priou, B. Rance
    and E. Kempf ([@kempf:hal-03519085]), and later refined by AP-HP's Data
    Science team.
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
