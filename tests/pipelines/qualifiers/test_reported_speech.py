from typing import List

from pytest import fixture, mark

from edsnlp.pipelines.qualifiers.reported_speech import ReportedSpeech
from edsnlp.utils.examples import parse_example

examples: List[str] = [
    # (
    #     'Pas de critique de sa TS de nov 2020 "je '
    #     "<ent reported_speech_=REPORTED>regrette</ent> d'avoir raté\"."
    # ),
    # "Décrit un scénario d'<ent reported_speech_=REPORTED>IMV</ent>",
    (
        "Elles sont décrites par X.x. comme des appels à l'aide "
        "« La <ent reported_speech_=REPORTED>pendaison</ent> "
        "a permis mon hospitalisation »."
    ),
    (
        "Rapporte une tristesse de l'humeur avec des idées "
        "<ent reported_speech_=REPORTED>suicidiares</ent> à "
        "type de pendaison,"
    ),
    (
        "Décrit un fléchissement thymique depuis environ "
        "1 semaine avec idées suicidaires scénarisées "
        "(<ent reported_speech_=REPORTED>intoxication "
        "médicamenteuse volontaire)</ent>"
    ),
    (
        "Dit ne pas savoir comment elle est tombé. "
        'Minimise la chute. Dit que "ça arrive. Badaboum". '
        "Dit ne pas avoir fait <ent reported_speech_=REPORTED>IMV</ent>."
    ),
    (
        "Le patient parle \"d'en finir\", et dit qu'il a pensé "
        "plusieurs fois à se pendre où à se faire une "
        "<ent reported_speech_=REPORTED>phlébotomie</ent> "
        "lorsqu'il était dans la rue, diminution de ces "
        "idées noires depuis qu'il vit chez son fils"
    ),
]

# A long test to check leakage from one entity to the next.
examples.append(
    "le patient est admis pour coronavirus. il dit qu'il n'est "
    "pas <ent reported_speech=True>malade</ent>.\n"
    "les tests sont positifs.\n"
    "il est <ent reported_speech=False>malade</ent>"
)


@fixture
def reported_speech_factory(blank_nlp):

    default_config = dict(
        pseudo=None,
        preceding=None,
        following=None,
        quotation=None,
        verbs=None,
        attr="NORM",
        within_ents=False,
        explain=True,
    )

    def factory(on_ents_only, **kwargs):

        config = dict(**default_config)
        config.update(kwargs)

        return ReportedSpeech(
            nlp=blank_nlp,
            on_ents_only=on_ents_only,
            **config,
        )

    return factory


@mark.parametrize("on_ents_only", [True, False])
def test_reported_speech(blank_nlp, reported_speech_factory, on_ents_only):

    reported_speech = reported_speech_factory(on_ents_only=on_ents_only)

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = reported_speech(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert getattr(ent._, modifier.key) == modifier.value

                if not on_ents_only:
                    for token in ent:
                        assert getattr(token._, modifier.key) == modifier.value


def test_reported_speech_within_ents(blank_nlp, reported_speech_factory):

    reported_speech = reported_speech_factory(on_ents_only=True, within_ents=True)

    examples = [
        "Le patient a une <ent reported_speech=True>"
        "fracture au tibias, il dit qu'il a mal</ent>."
    ]

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = reported_speech(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."
