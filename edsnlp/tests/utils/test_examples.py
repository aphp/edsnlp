from edsnlp.utils.examples import find_matches, parse_example, parse_match

example = (
    "Le <ent int_value=1 float_value=.3>patient</ent> "
    "n'est pas <ent polarity_=NEG negation=true>malade</ent>."
)


def test_find_matches():
    matches = find_matches(example=example)

    assert len(matches) == 2

    match1, match2 = matches

    assert match1.group() == "<ent int_value=1 float_value=.3>patient</ent>"
    assert match2.group() == "<ent polarity_=NEG negation=true>malade</ent>"


def test_parse_match():
    matches = find_matches(example=example)
    match = parse_match(matches[0])

    assert match.text == "patient"
    assert match.modifiers == "int_value=1 float_value=.3"


def test_parse_example():
    text, entities = parse_example(example=example)

    assert text == "Le patient n'est pas malade."

    entity1, entity2 = entities

    assert text[entity1.start_char : entity1.end_char] == "patient"
    assert text[entity2.start_char : entity2.end_char] == "malade"

    m1, m2, m3, m4 = entity1.modifiers + entity2.modifiers

    assert m1.key == "int_value"
    assert m1.value == 1

    assert m2.key == "float_value"
    assert m2.value == 0.3

    assert m3.key == "polarity_"
    assert m3.value == "NEG"

    assert m4.key == "negation"
    assert m4.value
