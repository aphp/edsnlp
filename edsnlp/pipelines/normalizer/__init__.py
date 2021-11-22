from functools import lru_cache
from typing import Union

from spacy.tokens import Doc, Span, Token

from .accents import Accents
from .endlines import EndLines, EndLinesModel
from .lowercase import Lowercase
from .normalizer import Normalizer
from .pollution import Pollution
from .quotes import Quotes

if not Token.has_extension("keep"):
    Token.set_extension("keep", default=True)


if not Token.has_extension("normalization"):
    Token.set_extension("normalization", default=None)


if not Doc.has_extension("normalized"):
    Doc.set_extension("normalized", default=None)


@lru_cache(maxsize=1)
def _norm2original(doc):
    return [token.i for token in doc if token._.keep] + [len(doc)]


@lru_cache(maxsize=1)
def _original2norm(doc):
    n2o = doc._.norm2original
    o2n = []

    for n, o in enumerate(n2o):
        if o == len(o2n):
            o2n.append(n)
        o2n.extend([(n - 1) for _ in range(o - len(o2n))])

    o2n.extend([n for _ in range(len(doc) - len(o2n))])

    o2n.append(len(doc._.normalized))

    return o2n


if not Doc.has_extension("norm2original"):
    Doc.set_extension("norm2original", getter=_norm2original)

if not Doc.has_extension("original2norm"):
    Doc.set_extension("original2norm", getter=_original2norm)


def _get_normalized_span(span: Span) -> Span:
    doc = span.doc
    start, end = span.start, span.end

    ns, ne = doc._.original2norm[start], doc._.original2norm[end]

    return doc._.normalized[ns:ne]


if not Span.has_extension("normalized"):
    Span.set_extension("normalized", getter=_get_normalized_span)

if not Token.has_extension("normalized"):
    Token.set_extension(
        "normalized",
        getter=lambda token: token.doc._.normalized[token.doc._.original2norm[token.i]],
    )
