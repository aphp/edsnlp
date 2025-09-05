from datetime import datetime

import spacy
from pytest import mark

from edsnlp.pipelines.qualifiers.history import History
from edsnlp.pipelines.qualifiers.history.patterns import history
from edsnlp.pipelines.terminations import termination

text = """COMPTE RENDU D'HOSPITALISATION du 11/07/2018 au 12/07/2018

MOTIF D'HOSPITALISATION
Monsieur Dupont 30\n2 Jean Michel, de sexe masculin, âgée de 39 ans,
née le 23/11/1978, est admis pour une toux.
Il a été hospitalisé du 11/08/2019 au 17/08/2019,
avec un antécédent d'asthme il y a 25 jours.

ANTÉCÉDENTS
Antécédents médicaux :
Premier épisode: il a été hospitalisé pour asthme cette semaine-ci,
il y a 3 jours, le 13 août 2020.
Hier, le patient est venu pour une toux dont les symptômes,
seraient apparus il y a 2 mois."""


@mark.parametrize("use_sections", [True, False])
@mark.parametrize("use_dates", [True, False])
@mark.parametrize("on_ents_only", [True, False])
@mark.parametrize("exclude_birthdate", [True, False])
def test_history(lang, use_sections, use_dates, exclude_birthdate, on_ents_only):
    nlp = spacy.blank(lang)
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe(
        "eds.matcher",
        config=dict(
            terms=dict(
                respiratoire=[
                    "asthmatique",
                    "asthme",
                    "toux",
                ]
            )
        ),
    )
    nlp.add_pipe(
        "eds.history", config=dict(use_sections=use_sections, use_dates=use_dates)
    )
    doc = nlp(text)
    nlp.remove_pipe("eds.history")
    nlp.add_pipe("eds.sections")
    nlp.add_pipe("eds.dates")
    doc = nlp(text)
    doc._.note_datetime = datetime(2020, 8, 11)
    doc._.birth_datetime = datetime(1978, 11, 23)

    history_nlp = History(
        nlp=nlp,
        attr="NORM",
        history=history,
        termination=termination,
        use_sections=use_sections,
        use_dates=use_dates,
        exclude_birthdate=exclude_birthdate,
        closest_dates_only=True,
        history_limit=15,
        explain=True,
        on_ents_only=on_ents_only,
    )
    doc = history_nlp(doc)

    if use_dates:
        assert doc.ents[0]._.history is not exclude_birthdate
        if not exclude_birthdate:
            assert doc.ents[0]._.history_cues[0].text == "23/11/1978"

    assert doc.ents[1]._.history and doc.ents[1]._.history_cues[0].label_ == "history"

    if use_sections:
        assert doc.ents[2]._.history is not use_dates
        assert doc.ents[2]._.history_cues[0].label_ == "ATCD"
        if use_dates:
            assert doc.ents[2]._.recent_cues[0].label_ == "relative_date"
