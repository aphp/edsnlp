from pytest import mark

from edsnlp.utils.examples import parse_example


@mark.parametrize(
    "config",
    [
        dict(matcher="regex", attr="TEXT"),
        dict(matcher="phrase", attr="NORM"),
        dict(matcher="phrase", attr="TEXT"),
    ],
)
def test_context_matcher(blank_nlp, config):

    blank_nlp.add_pipe("eds.normalizer")
    blank_nlp.add_pipe("eds.context-matcher", config=config)

    example = (
        "Le patient s'appelle <ent label=FIRST>Jean-Michel</ent> "
        "<ent label=LAST>Test</ent>, il est malade."
    )
    context = dict(FIRST=["Jean-Michel"], LAST=["Test"])

    text, entities = parse_example(example)

    for doc, _ in blank_nlp.pipe([(text, context)], as_tuples=True):

        ents = list(doc.ents)

        assert len(ents) == len(entities)

        for ent, entity in zip(doc.ents, entities):
            assert ent.label_ == entity.modifiers[0].value
            assert ent.text == text[entity.start_char : entity.end_char]
