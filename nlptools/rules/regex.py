import re
from typing import Optional, List

from ..utils.spacy import span_from_span

from loguru import logger


class RegexMatcher(object):

    def __init__(self, alignment_mode: Optional[str] = 'expand'):
        self.alignment_mode = alignment_mode
        self.regex = dict()

    def add(self, key: str, patterns: List[str]):
        self.regex[key] = [re.compile(pattern) for pattern in patterns]

    def remove(self, key: str):
        del self.regex[key]

    def match(self, doc):
        for key, patterns in self.regex.items():
            for pattern in patterns:
                for match in pattern.finditer(doc.text):
                    logger.trace(f'Matched a regex from {key}: {repr(match.group())}')
                    span = doc.char_span(
                        match.start(),
                        match.end(),
                        label=key,
                        alignment_mode=self.alignment_mode,
                    )
                    if span is not None:
                        yield span

    def __call__(self, doc):
        for match in self.match(doc):
            yield match
