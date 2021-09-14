from edsnlp.utils.labeltool import docs2labeltool
from labeltool.labelling import Labels, Labelling

from ipywidgets import Output
from IPython.display import display


texts = [
    "Le patient est malade",
    "Le patient n'est pas malade",
    "Le patient est peut-être malade",
    "Le patient dit qu'il est malade",
]


def test_docs2labeltool(nlp):

    modifiers = ["negated", "hypothesis", "reported_speech"]

    docs = list(nlp.pipe(texts))
    df = docs2labeltool(docs, extensions=modifiers)

    labels = Labels()

    for label in df.label_name.unique():

        labels.add(
            name=label,
            selection_type="text",
        )

    labeller = Labelling(
        df,
        modifiers=modifiers,
        labels_dict=labels.dict,
        out=Output(),
        display=display,
    )
    labeller.run()
