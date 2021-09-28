from typing import List
from edsnlp.utils.examples import parse_example

from pytest import fixture, mark

from edsnlp.pipelines.rspeech import ReportedSpeech, terms

examples: List[str] = [
    """Pas de critique de sa TS de nov 2020 "je <ent reported_speech_=REPORTED>regrette</ent> d'avoir raté".""",
    "Décrit un scénario d'<ent reported_speech_=REPORTED>IMV</ent>",
    "Elles sont décrites par X.x. comme des appels à l’aide « La <ent reported_speech_=REPORTED>pendaison</ent> a permis mon hospitalisation ».",
    "Rapporte une tristesse de l'humeur avec des idées <ent reported_speech_=REPORTED>suicidiares</ent> à type de pendaison,",
    "Décrit un fléchissement thymique depuis environ 1 semaine avec idées suicidaires scénarisées (<ent reported_speech_=REPORTED>intoxication médicamenteuse volontaire)</ent>",
    """Dit ne pas savoir comment elle est tombé. Minimise la chute. Dit que "ça arrive. Badaboum". Dit ne pas avoir fait <ent reported_speech_=REPORTED>IMV</ent>.""",
    """Le patient parle "d'en finir", et dit qu'il a pensé plusieurs fois à se pendre où à se faire une <ent reported_speech_=REPORTED>phlébotomie</ent> lorsqu'il était dans la rue, diminution de ces idées noires depuis qu'il vit chez son fils""",
]


@fixture
def rspeech_factory(blank_nlp):

    default_config = dict(
        preceding=terms.preceding,
        following=terms.following,
        verbs=terms.verbs,
        quotation=terms.quotation,
        fuzzy=False,
        filter_matches=False,
        attr="LOWER",
        fuzzy_kwargs=None,
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
def test_rspeech(blank_nlp, rspeech_factory, on_ents_only):

    rspeech = rspeech_factory(on_ents_only=on_ents_only)

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)
        doc.ents = [
            doc.char_span(ent.start_char, ent.end_char, label="ent") for ent in entities
        ]

        doc = rspeech(doc)

        for entity, ent in zip(entities, doc.ents):

            for modifier in entity.modifiers:

                assert (
                    getattr(ent._, modifier.key) == modifier.value
                ), f"{modifier.key} labels don't match."

                if not on_ents_only:
                    for token in ent:
                        assert (
                            getattr(token._, modifier.key) == modifier.value
                        ), f"{modifier.key} labels don't match."
