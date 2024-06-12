from edsnlp.utils.examples import parse_example

examples = [
    "J'ai vu le patient à cause d'une <ent>TS</ent> médicamenetuse."
    "J'ai vu le patient à cause d'une ts médicamenetuse.",
    "J'ai vu le patient à cause d'une <ent>IMV</ent>.",
    "surface TS",
    "Patiente hospitalisée à cause d'une <ent>Tentative d'autolyse</ent>.",
    "Le patient exprime des idées de défenestration",
    "vu aux urgences suite à une <ent>défenestration volontaire</ent>",
    "amené par les pompiers à cause d'une <ent>phlebotomie</ent>",
    "Antécédents :\n- <ent>pendaison</ent> (2010)",
    "copain du patient : plusieurs événements d'<ent>autodestruction</ent>",
    "suspicion d'<ent>ingestion de caustique</ent> avec des idées suicidaires",
]


def test_suicide_attempt(blank_nlp):
    blank_nlp.add_pipe("eds.suicide_attempt")

    for text, entities in map(parse_example, examples):
        doc = blank_nlp(text)

        assert len(doc.ents) == len(entities)

        for ent, entity in zip(doc.ents, entities):
            assert ent.text == text[entity.start_char : entity.end_char]
