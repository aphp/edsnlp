from typing import Union

from spacy.tokens import Span

from edsnlp.matchers.utils import get_text

DIGITS_VALUE = list(range(11))
DIGITS_STR = [
    ["zero"],
    ["un", "une", "i"],
    ["deux", "ii"],
    ["trois", "iii"],
    ["quatre", "iv"],
    ["cinq", "v"],
    ["six", "vi"],
    ["sept", "vii"],
    ["huit", "viii"],
    ["neuf", "ix"],
    ["dix", "x"],
]

DIGITS_MAPPINGS = {
    string: digit for digit, strings in enumerate(DIGITS_STR) for string in strings
}


def parse_digit(s: Union[str, Span], **kwargs):
    if isinstance(s, Span):
        string = get_text(
            s,
            attr=kwargs.get("attr", "TEXT"),
            ignore_excluded=kwargs.get("ignore_excluded", True),
        )
    else:
        string = s
    string = string.lower().strip()
    try:
        return int(string)
    except ValueError:
        parsed = DIGITS_MAPPINGS.get(string, None)
        return parsed
