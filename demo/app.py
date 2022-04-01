from typing import Any
import pandas as pd
import spacy
import streamlit as st
from spacy import displacy
from spacy.tokens import Span

DEFAULT_TEXT = """\
Motif :
Le patient est admis le 29 août pour des difficultés respiratoires.

Antécédents familiaux :
Le père du patient n'est pas asthmatique.

HISTOIRE DE LA MALADIE
Le patient dit avoir de la toux depuis trois jours. \
Elle a empiré jusqu'à nécessiter un passage aux urgences.
A noter deux petits kystes bénins de 1 et 2cm biopsiés en 2005.

Priorité: 2 (établie par l'IAO à l'entrée)

Conclusion
Possible infection au coronavirus\
"""

REGEX = """
# RegEx and terms matcher
nlp.add_pipe(
    "eds.matcher",
    config=dict(
        regex=dict(custom=r"{custom_regex}"),
        attr="NORM",
    ),
)
"""

CODE = """
import spacy

# Declare the pipeline
nlp = spacy.blank("fr")

# General-purpose components
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sentences")
{pipes}
# Qualifier pipelines
nlp.add_pipe("eds.negation")
nlp.add_pipe("eds.family")
nlp.add_pipe("eds.hypothesis")
nlp.add_pipe("eds.reported_speech")

# Define the note text
text = {text}

# Apply the pipeline
doc = nlp(text)

# Explore matched elements
doc.ents
"""


@st.cache(allow_output_mutation=True)
def load_model(
    covid: bool,
    dates: bool,
    measures: bool,
    charlson: bool,
    sofa: bool,
    priority: bool,
    custom_regex: str,
):

    pipes = []

    # Declare the pipeline
    nlp = spacy.blank("fr")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")

    if covid:
        nlp.add_pipe("eds.covid")
        pipes.append('nlp.add_pipe("eds.covid")')

    if dates:
        nlp.add_pipe("eds.dates")
        pipes.append('nlp.add_pipe("eds.dates")')

    if measures:
        nlp.add_pipe("eds.measures")
        pipes.append('nlp.add_pipe("eds.measures")')

    if charlson:
        nlp.add_pipe("eds.charlson")
        pipes.append('nlp.add_pipe("eds.charlson")')

    if sofa:
        nlp.add_pipe("eds.SOFA")
        pipes.append('nlp.add_pipe("eds.SOFA")')

    if priority:
        nlp.add_pipe("eds.emergency.priority")
        pipes.append('nlp.add_pipe("eds.emergency.priority")')

    if pipes:
        pipes.insert(0, "# Entity extraction pipelines")

    if custom_regex:
        nlp.add_pipe(
            "eds.matcher",
            config=dict(
                regex=dict(custom=custom_regex),
                attr="NORM",
            ),
        )

        regex = REGEX.format(custom_regex=custom_regex)

    else:
        regex = ""

    nlp.add_pipe("eds.negation")
    nlp.add_pipe("eds.family")
    nlp.add_pipe("eds.hypothesis")
    nlp.add_pipe("eds.reported_speech")

    return nlp, pipes, regex


st.set_page_config(
    page_title="EDS-NLP Demo",
    page_icon="📄",
)

st.title("EDS-NLP")

st.warning(
    "You should **not** put sensitive data in the example, as this application "
    "**is not secure**."
)

st.sidebar.header("About")
st.sidebar.markdown(
    "EDS-NLP is a contributive effort maintained by AP-HP's Data Science team. "
    "Have a look at the "
    "[documentation](https://aphp.github.io/edsnlp/) for "
    "more information on the available pipelines."
)

st.sidebar.header("Pipeline")
st.sidebar.markdown(
    "This example runs a simplistic pipeline detecting a few synonyms for "
    "COVID-related entities.\n\n"
    "You can add or remove pre-defined pipeline components, and see how "
    "the pipeline reacts. You can also search for your own custom RegEx."
)

st.sidebar.header("Custom RegEx")
custom_regex = st.sidebar.text_input(
    "Regular Expression:",
    r"asthmatique|difficult[ée]s?\srespiratoires?",
)
st.sidebar.markdown("The RegEx you defined above is detected under the `custom` label.")

st.sidebar.subheader("Pipeline Components")
covid = st.sidebar.checkbox("COVID", value=True)
dates = st.sidebar.checkbox("Dates", value=True)
measures = st.sidebar.checkbox("Measures", value=True)
priority = st.sidebar.checkbox("Emergency Priority Score", value=True)
charlson = st.sidebar.checkbox("Charlson Score", value=True)
sofa = st.sidebar.checkbox("SOFA Score", value=True)
st.sidebar.markdown(
    "These are just a few of the pipelines provided out-of-the-box by EDS-NLP. "
    "See the [documentation](https://aphp.github.io/edsnlp/latest/pipelines/) "
    "for detail."
)

model_load_state = st.info("Loading model...")

nlp, pipes, regex = load_model(
    covid=covid,
    dates=dates,
    measures=measures,
    charlson=charlson,
    sofa=sofa,
    priority=priority,
    custom_regex=custom_regex,
)

model_load_state.empty()

st.header("Enter a text to analyse:")
text = st.text_area(
    "Modify the following text and see the pipeline react :",
    DEFAULT_TEXT,
    height=375,
)

doc = nlp(text)

st.header("Visualisation")

st.markdown(
    "The pipeline extracts simple entities using a dictionnary of RegEx (see the "
    "[Export the pipeline section](#export-the-pipeline) for more information)."
)

ents = list(doc.ents)

for ent in ents:
    if ent._.score_value:
        ent._.value = ent._.score_value

for date in doc.spans.get("dates", []):
    span = Span(doc, date.start, date.end, label="date")
    span._.value = span._.date
    ents.append(span)

for measure in doc.spans.get("measures", []):
    span = Span(doc, measure.start, measure.end, label=measure.label_)
    span._.value = span._.value
    ents.append(span)


doc.ents = ents

category20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]

labels = [
    "date",
    "covid",
    "eds.emergency.priority",
    "eds.SOFA",
    "eds.charlson",
    "eds.measures.size",
    "eds.measures.weight",
    "eds.measures.angle",
]

colors = {label: cat for label, cat in zip(labels, category20)}
colors["custom"] = "linear-gradient(90deg, #aa9cfc, #fc9ce7)"
options = {
    "colors": colors,
}

html = displacy.render(doc, style="ent", options=options)
html = html.replace("line-height: 2.5;", "line-height: 2.25;")
html = (
    '<div style="padding: 10px; border: solid 2px; border-radius: 10px; '
    f'border-color: #afc6e0;">{html}</div>'
)
st.write(html, unsafe_allow_html=True)

data = []
for ent in doc.ents:

    if ent.label_ == "date" or "measure" in ent.label_:
        d = dict(
            start=ent.start_char,
            end=ent.end_char,
            lexical_variant=ent.text,
            label=ent.label_,
            negation="",
            family="",
            hypothesis="",
            reported_speech="",
        )

    else:
        d = dict(
            start=ent.start_char,
            end=ent.end_char,
            lexical_variant=ent.text,
            label=ent.label_,
            negation="YES" if ent._.negation else "NO",
            family="YES" if ent._.family else "NO",
            hypothesis="YES" if ent._.hypothesis else "NO",
            reported_speech="YES" if ent._.reported_speech else "NO",
        )

    try:
        d["normalized_value"] = str(ent._.value)
    except TypeError:
        d["normalized_value"] = ""

    data.append(d)

st.header("Entity qualification")


def color_qualifiers(val: Any) -> str:
    """
    Add color to qualifiers.

    Parameters
    ----------
    val : Any
        DataFrame value

    Returns
    -------
    str
        style
    """
    if val == "NO":
        return "color: #dc3545;"
    elif val == "YES":
        return "color: #198754;"
    return ""


if data:
    df = pd.DataFrame.from_records(data)
    df.normalized_value = df.normalized_value.replace({"None": ""})

    df = df.style.applymap(color_qualifiers)

    st.dataframe(df)

else:
    st.markdown("You pipeline did not match any entity...")

pipes_text = ""

if pipes:
    pipes_text += "\n" + "\n".join(pipes) + "\n"
if regex:
    pipes_text += regex

code = CODE.format(
    pipes=pipes_text,
    text=f'"""\n{text}\n"""',
)

st.header("Export the pipeline")
st.markdown(
    "The code below recreates the pipeline. Copy and paste it "
    "in a Jupyter Notebook to interact with it."
)
with st.expander("Show the runnable code"):
    st.markdown(f"```python\n{code}\n```\n\nThis code runs as is.")
