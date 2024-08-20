import spacy

import edsnlp


def make_nlp(lang):
    if lang == "eds":
        model = spacy.blank("eds")
    else:
        model = edsnlp.blank("fr")

    model.add_pipe("eds.normalizer")

    model.add_pipe("eds.sentences")
    model.add_pipe("eds.sections")

    model.add_pipe(
        "eds.matcher",
        config=dict(
            terms=dict(patient="patient"),
            attr="NORM",
            ignore_excluded=True,
        ),
    )
    model.add_pipe(
        "eds.matcher",
        name="matcher2",
        config=dict(
            regex=dict(anomalie=r"anomalie"),
        ),
    )

    model.add_pipe("eds.hypothesis")
    model.add_pipe("eds.negation")
    model.add_pipe("eds.family")
    model.add_pipe("eds.history")
    model.add_pipe("eds.reported_speech")

    model.add_pipe("eds.dates")
    model.add_pipe("eds.quantities")

    return model
