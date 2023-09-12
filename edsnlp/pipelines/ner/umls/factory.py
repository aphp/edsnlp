from typing import Any, Dict, Union

from spacy.language import Language
from typing_extensions import Literal

from edsnlp.pipelines.core.terminology.terminology import TerminologyMatcher

from ...base import SpanSetterArg
from .patterns import get_patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    ignore_space_tokens=False,
    term_matcher="exact",
    term_matcher_config={},
    pattern_config=dict(languages=["FRE"], sources=None),
    label="umls",
    span_setter={"ents": True, "umls": True},
)


@Language.factory(
    "eds.umls",
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str = "eds.umls",
    *,
    attr: Union[str, Dict[str, str]] = "NORM",
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    term_matcher: Literal["exact", "simstring"] = "exact",
    term_matcher_config: Dict[str, Any] = {},
    pattern_config: Dict[str, Any] = dict(languages=["FRE"], sources=None),
    label: str = "umls",
    span_setter: SpanSetterArg = {"ents": True, "umls": True},
):
    """
    The `eds.umls` pipeline component matches the UMLS (Unified Medical Language System
    from NIH) terminology.

    !!! warning "Very low recall"

        When using the `exact` matching mode, this component has a very poor recall
        performance. We can use the `simstring` mode to retrieve approximate matches,
        albeit at the cost of a significantly higher computation time.

    Examples
    --------
    `eds.umls` is an additional module that needs to be setup by:

    1. `pip install -U umls_downloader`
    2. [Signing up](https://uts.nlm.nih.gov/uts/signup-login) for a UMLS Terminology
       Services Account. After filling a short form, you will receive your token API
       within a few days.
    3. Set `UMLS_API_KEY` locally: `export UMLS_API_KEY=your_api_key`

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.umls")

    text = "Grosse toux: le malade a été mordu par des Amphibiens " "sous le genou"

    doc = nlp(text)

    doc.ents
    # Out: (toux, a, par, Amphibiens, genou)

    ent = doc.ents[0]

    ent.label_
    # Out: umls

    ent._.umls
    # Out: C0010200
    ```

    You can easily change the default languages and sources with the `pattern_config`
    argument:

    ```python
    import spacy

    # Enable the French and English languages, through the French MeSH and LOINC
    pattern_config = dict(languages=["FRE", "ENG"], sources=["MSHFRE", "LNC"])

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.umls", config=dict(pattern_config=pattern_config))
    ```

    See more options of languages and sources
    [here](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/index.html).

    Parameters
    ----------
    nlp : Language
        spaCy `Language` object.
    name : str
        The name of the pipe
    attr : Union[str, Dict[str, str]]
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    ignore_excluded : bool
        Whether to skip excluded tokens during matching.
    ignore_space_tokens : bool
        Whether to skip space tokens during matching.
    term_matcher : TerminologyTermMatcher
        The term matcher to use, either "exact" or "simstring"
    term_matcher_config : Dict[str, Any]
        The configuration for the term matcher
    pattern_config : Dict[str, Any]
        The pattern retriever configuration
    label : str
        Label name to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Authors and citation
    --------------------
    The `eds.umls` pipeline was developed by AP-HP's Data Science team and INRIA
    SODA's team.
    """
    return TerminologyMatcher(
        nlp=nlp,
        name=name,
        regex=dict(),
        terms=get_patterns(pattern_config),
        attr=attr,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        term_matcher=term_matcher,
        term_matcher_config=term_matcher_config,
        label=label,
        span_setter=span_setter,
    )
