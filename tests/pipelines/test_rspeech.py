from typing import List
from edsnlp.utils.examples import parse_example

rpseech_examples: List[str] = [
    """Pas de critique de sa TS de nov 2020 "je <ent reported_speech_=REPORTED>regrette</ent> d'avoir raté".""",
    "Décrit un scénario d'<ent reported_speech_=REPORTED>IMV</ent>",
    "Elles sont décrites par X.x. comme des appels à l’aide « La <ent reported_speech_=REPORTED>pendaison</ent> a permis mon hospitalisation ».",
    "Rapporte une tristesse de l'humeur avec des idées <ent reported_speech_=REPORTED>suicidiares</ent> à type de pendaison,",
    "Décrit un fléchissement thymique depuis environ 1 semaine avec idées suicidaires scénarisées (<ent reported_speech_=REPORTED>intoxication médicamenteuse volontaire)</ent>",
    """Dit ne pas savoir comment elle est tombé. Minimise la chute. Dit que "ça arrive. Badaboum". Dit ne pas avoir fait <ent reported_speech_=REPORTED>IMV</ent>.""",
    """Le patient parle "d'en finir", et dit qu'il a pensé plusieurs fois à se pendre où à se faire une <ent reported_speech_=REPORTED>phlébotomie</ent> lorsqu'il était dans la rue, diminution de ces idées noires depuis qu'il vit chez son fils""",
]


def test_rspeech(nlp):

    for example in rpseech_examples:
        text, entities = parse_example(example=example)

        doc = nlp(text)

        for ent in entities:

            span = doc.char_span(ent.start_char, ent.end_char)

            for modifier in ent.modifiers:

                assert all(
                    [getattr(token._, modifier.key) == modifier.value for token in span]
                ), f"{modifier.key} labels don't match."
