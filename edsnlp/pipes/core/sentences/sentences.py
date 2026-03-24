import warnings
from typing import List, Optional, Tuple

from spacy.tokens import Doc

from edsnlp.core import PipelineProtocol

from ...base import BaseComponent
from .fast_sentences import FastSentenceSegmenter
from .terms import punctuation

# Default punctuation defined for the sentencizer : https://spacy.io/api/sentencizer
# fmt: off
DEFAULT_BULLET_STARTERS = [
    "-", "_", "*", "•", "·", "", "⁃", "‣", "⁎", "⁑", "+",
    "→", "⇒", "⇨", "➔", "➜", "➝", "➞", "➟", "➠", "➡", "➡️",
]
# fmt: on


def generate_capitalized_shapes(
    upper_min: int = 2,
    upper_max: int = 13,
    x_min: int = 2,
    x_max: int = 12,
    include_all_caps: bool = True,
    include_titlecase: bool = True,
    include_apostrophe: bool = True,
) -> Tuple[str, ...]:
    """
    Generate spaCy `token.shape_` patterns used to detect capitalized line starts.
    """
    shapes: List[str] = []

    if include_all_caps:
        for i in range(upper_min, upper_max + 1):
            shapes.append("X" * i)

    if include_titlecase:
        for i in range(x_min, x_max + 1):
            shapes.append("X" + "x" * (i - 1))

    if include_apostrophe:
        shapes.append("X'")

    return tuple(shapes)


DEFAULT_CAPITALIZED_SHAPES: Tuple[str, ...] = generate_capitalized_shapes(
    upper_min=2,
    upper_max=13,
    x_min=2,
    x_max=12,
    include_apostrophe=True,
)

LEGACY_CAPITALIZED_SHAPES: Tuple[str, ...] = ("X'", "Xx", "Xxx", "Xxxx", "Xxxxx")


class SentenceSegmenter(BaseComponent):
    r'''
    The `eds.sentences` matcher provides an alternative to spaCy's default
    `sentencizer`, aiming to overcome some of its limitations.

    Indeed, the `sentencizer` merely looks at period characters to detect the end of a
    sentence, a strategy that often fails in a clinical note settings. Our
    `eds.sentences` component also classifies end-of-lines as sentence boundaries if
    the subsequent token begins with an uppercase character, leading to slightly better
    performances. It can additionally leverage expanded capitalization patterns and
    bullet-like list starters, which are frequent in structured medical documents.

    Moreover, the `eds.sentences` component use the output of the `eds.normalizer`
    and `eds.endlines` output by default when these components are added to the
    pipeline.

    Examples
    --------
    === "EDS-NLP"

        ```{ .python .no-check }
        import edsnlp, edsnlp.pipes as eds

        nlp = edsnlp.blank("eds")
        nlp.add_pipe(eds.sentences())  # same as nlp.add_pipe("eds.sentences")

        text = """Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        Il lui était arrivé la même chose il y a deux ans."
        """

        doc = nlp(text)

        for sentence in doc.sents:
            print("<s>", sentence, "</s>")
        # Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        # Out:  <\s>
        # Out: <s> Il lui était arrivé la même chose il y a deux ans. <\s>
        ```

    === "spaCy sentencizer"

        ```{ .python .no-check }
        import edsnlp, edsnlp.pipes as eds

        nlp = edsnlp.blank("eds")
        nlp.add_pipe("sentencizer")

        text = """Le patient est admis le 23 août 2021 pour une douleur à l'estomac"
        Il lui était arrivé la même chose il y a deux ans.
        """

        doc = nlp(text)

        for sentence in doc.sents:
            print("<s>", sentence, "</s>")
        # Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        # Out: Il lui était arrivé la même chose il y a deux ans. <\s>
        ```

    Notice how EDS-NLP's implementation is more robust to ill-defined sentence endings.


    Parameters
    ----------
    nlp: PipelineProtocol
        The EDS-NLP pipeline
    name: Optional[str]
        The name of the component
    punct_chars: Optional[List[str]]
        Punctuation characters.
    use_endlines: bool
        Whether to use endlines prediction.
    ignore_excluded: bool
        Whether to ignore excluded tokens.
    check_capitalized: bool
        Whether to check for capitalized words after newlines or full stops.
    capitalized_mode : Optional[str], {"legacy", "expanded"}, default "expanded"
        Selects the preset of capitalized shapes used when `check_capitalized=True`
        and no explicit `capitalized_shapes` are provided.
    capitalized_shapes: Optional[List[str]]
        Capitalized shapes.
    min_newline_count: int
        The minimum number of newlines to consider a newline-triggered sentence.
    hard_newline_count: int | None
        The minimum number of consecutive newlines to force a sentence boundary,
        independently of capitalization. Use `None` to disable this rule.
    use_bullet_start: bool
        Whether to check for bullet starters after newlines or full stops.
    bullet_starters: Optional[List[str]]
        Bullet starters characters.

    Authors and citation
    --------------------
    The `eds.sentences` component was developed by AP-HP's Data Science team.
    '''

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: Optional[str] = "sentences",
        punct_chars: Optional[List[str]] = None,
        use_endlines: Optional[bool] = None,
        ignore_excluded: bool = True,
        check_capitalized: bool = True,
        capitalized_mode: Optional[str] = "expanded",
        capitalized_shapes: Optional[List[str]] = None,
        min_newline_count: int = 1,
        hard_newline_count: Optional[int] = None,
        use_bullet_start: bool = False,
        bullet_starters: Optional[List[str]] = None,
    ):
        super().__init__(nlp, name)
        if (
            min_newline_count > 1
            or (hard_newline_count is not None and hard_newline_count > 1)
        ) and nlp.lang != "eds":
            warnings.warn(
                "To use newline thresholds > 1, you need to use the 'eds' language "
                "to split newlines into single tokens (e.g. `edsnlp.blank('eds')`)."
            )

        if punct_chars is None:
            punct_chars = punctuation

        if check_capitalized and capitalized_shapes is None:
            capitalized_shapes = (
                LEGACY_CAPITALIZED_SHAPES
                if capitalized_mode == "legacy"
                else DEFAULT_CAPITALIZED_SHAPES
            )
        capitalized_shapes = tuple(capitalized_shapes or ())

        if bullet_starters is None:
            bullet_starters = DEFAULT_BULLET_STARTERS

        self.fast_segmenter = FastSentenceSegmenter(
            vocab=nlp.vocab,
            punct_chars=punct_chars,
            use_endlines=use_endlines,
            ignore_excluded=ignore_excluded,
            check_capitalized=check_capitalized,
            capitalized_shapes=capitalized_shapes,
            min_newline_count=min_newline_count,
            hard_newline_count=(
                -1 if hard_newline_count is None else hard_newline_count
            ),
            use_bullet_start=use_bullet_start,
            bullet_starters=bullet_starters,
        )

    def __call__(self, doc: Doc):
        return self.fast_segmenter(doc)
