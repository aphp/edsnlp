import json
import re
from typing import Any, Dict, List, Tuple, Union

import regex
from pydantic import BaseModel, validator


class Match(BaseModel):
    start_char: int
    end_char: int
    text: str
    modifiers: str


class Modifier(BaseModel):
    key: str
    value: Union[int, float, bool, str, Dict[str, Any]]

    @validator("value")
    def optional_dict_parsing(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v.replace("'", '"'))
            except json.JSONDecodeError:
                return v
        return v


class Entity(BaseModel):
    start_char: int
    end_char: int
    modifiers: List[Modifier]


entity_pattern = re.compile(r"(<ent[^<>]*>[^<>]+</ent>)", flags=re.DOTALL)
text_pattern = re.compile(r"<ent.*>(.+)</ent>", flags=re.DOTALL)
modifiers_pattern = re.compile(r"<ent\s?(.*)>.+</ent>", flags=re.DOTALL)
single_modifiers_pattern = regex.compile(
    r"(?P<key>[^\s]+?)=((?P<value>{.*?})|(?P<value>[^\s']+)|'(?P<value>.+?)')",
    flags=regex.DOTALL,
)


def find_matches(example: str) -> List[re.Match]:
    """
    Finds entities within the example.

    Parameters
    ----------
    example : str
        Example to process.

    Returns
    -------
    List[re.Match]
        List of matches for entities.
    """
    return list(entity_pattern.finditer(example))


def parse_match(match: re.Match) -> Match:
    """
    Parse a regex match representing an entity.

    Parameters
    ----------
    match : re.Match
        Match for an entity.

    Returns
    -------
    Match
        Usable representation for the entity match.
    """

    lexical_variant = match.group()
    start_char = match.start()
    end_char = match.end()

    text = text_pattern.findall(lexical_variant)[0]
    modifiers = modifiers_pattern.findall(lexical_variant)[0]

    m = Match(start_char=start_char, end_char=end_char, text=text, modifiers=modifiers)

    return m


def parse_example(example: str) -> Tuple[str, List[Entity]]:
    """
    Parses an example : finds examples and removes the tags.

    Parameters
    ----------
    example : str
        Example to process.

    Returns
    -------
    Tuple[str, List[Entity]]
        Cleaned text and extracted entities.
    """

    matches = [parse_match(match) for match in find_matches(example=example)]
    text = ""
    entities = []

    cursor = 0

    for match in matches:

        text += example[cursor : match.start_char]
        start_char = len(text)
        text += match.text
        end_char = len(text)

        cursor = match.end_char

        entity = Entity(
            start_char=start_char,
            end_char=end_char,
            modifiers=[
                Modifier.parse_obj(m.groupdict())
                for m in single_modifiers_pattern.finditer(match.modifiers)
            ],
        )

        entities.append(entity)

    text += example[cursor:]

    return text, entities
