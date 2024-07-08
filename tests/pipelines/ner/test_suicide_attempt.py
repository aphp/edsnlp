from edsnlp.utils.examples import Entity, parse_example

examples = [
    "J'ai vu le patient à cause d'une <ent modality=suicide_attempt_unspecific>TS</ent> médicamenetuse."  # noqa: E501
    "J'ai vu le patient à cause d'une ts médicamenetuse.",  # noqa: E501
    "J'ai vu le patient à cause d'une <ent modality=intentional_drug_overdose>IMV</ent>.",  # noqa: E501
    "surface TS",  # noqa: E501
    "Patiente hospitalisée à cause d'une <ent modality=autolysis>Tentative d'autolyse</ent>.",  # noqa: E501
    "Le patient exprime des idées de défenestration",  # noqa: E501
    "vu aux urgences suite à une <ent modality=jumping_from_height>défenestration volontaire</ent>",  # noqa: E501
    "amené par les pompiers à cause d'une <ent modality=cuts>phlebotomie</ent>",  # noqa: E501
    "Antécédents :\n- <ent modality=strangling>pendaison</ent> (2010)",  # noqa: E501
    "copain du patient : plusieurs événements d'<ent modality=self_destructive_behavior>autodestruction</ent>",  # noqa: E501
    "suspicion d'<ent modality=burn_gas_caustic>ingestion de caustique</ent> avec des idées suicidaires",  # noqa: E501
]


def test_suicide_attempt(blank_nlp):
    blank_nlp.add_pipe("eds.suicide_attempt")

    for text, entities in map(parse_example, examples):
        doc = blank_nlp(text)

        assert len(doc.ents) == len(entities)

        for ent, entity in zip(doc.ents, entities):
            entity: Entity
            assert ent.text == text[entity.start_char : entity.end_char]
            assert ent._.suicide_attempt_modality == entity.modifiers_dict.get(
                "modality"
            )
