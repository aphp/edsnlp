import pandas as pd
import spacy
import streamlit as st
from spacy import displacy

DEFAULT_TEXT = """\
Motif :
Le patient est admis le 29 août pour des difficultés respiratoires.

Antécédents familiaux :
Le père est asthmatique, sans traitement particulier.

HISTOIRE DE LA MALADIE
Le patient dit avoir de la toux depuis trois jours. \
Elle a empiré jusqu'à nécessiter un passage aux urgences.

Conclusion
Possible infection au coronavirus\
"""

CODE = r"""
import spacy

# Declare the pipeline
nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")

nlp.add_pipe(
    "eds.matcher",
    config=dict(
        regex=dict(
            covid=r"(infection\s+au\s+)?(covid(-|\s+)?(19)?|corona(-|\s+)?virus)",
            traitement=r"traitements?|medicaments?",
            respiratoire=r"(difficult[eé]s?\s+respiratoires|asthmatique|toux)",
            {custom_label}=r"{custom_regex}"
        ),
        attr="LOWER",
    ),
)

{pipes}

# Define the note text
text = {text}

# Apply the pipeline
doc = nlp(text)

# Explore matched elements
doc.ents
"""


@st.cache(allow_output_mutation=True)
def load_model(
    negation,
    family,
    hypothesis,
    reported_speech,
    custom_label="CUSTOM",
    custom_regex=None,
):

    pipes = []

    nlp = spacy.blank("fr")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")

    regex = dict(
        covid=r"(infection\s+au\s+)?(covid(-|\s+)?(19)?|corona(-|\s+)?virus)",
        traitement=r"traitements?|medicaments?",
        respiratoire=r"(difficult[eé]s?\s+respiratoires|asthmatique|toux)",
    )

    if custom_regex:
        regex[custom_label] = [custom_regex]

    nlp.add_pipe(
        "eds.matcher",
        config=dict(
            regex=regex,
            attr="LOWER",
        ),
    )

    if negation:
        nlp.add_pipe("eds.negation")
        pipes.append('nlp.add_pipe("eds.negation")')

    if family:
        nlp.add_pipe("eds.family")
        pipes.append('nlp.add_pipe("eds.family")')

    if hypothesis:
        nlp.add_pipe("eds.hypothesis")
        pipes.append('nlp.add_pipe("eds.hypothesis")')

    if reported_speech:
        nlp.add_pipe("eds.reported_speech")
        pipes.append('nlp.add_pipe("eds.reported_speech")')

    return nlp, pipes


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
    "[documentation](https://datasciencetools-pages.eds.aphp.fr/edsnlp/) for "
    "more information on the available pipelines."
)

st.sidebar.header("Pipeline")
st.sidebar.markdown(
    "This example runs a simplistic pipeline detecting a few synonyms for "
    "COVID-related entities.\n\nYou can add or remove qualifiers, and see how "
    "the pipeline reacts. You can also search for your own custom RegEx."
)

# st.sidebar.header("Custom RegEx")
# custom_label = st.sidebar.text_input("Label:", r"custom")
custom_label = "CUSTOM"
custom_regex = st.sidebar.text_input(
    "Regular Expression:",
    r"passage\s\w+\surgences",
)

st.sidebar.subheader("Qualifiers")
negation = st.sidebar.checkbox("Negation", value=True)
family = st.sidebar.checkbox("Family", value=True)
hypothesis = st.sidebar.checkbox("Hypothesis", value=True)
reported_speech = st.sidebar.checkbox("Repoted Speech", value=True)

model_load_state = st.info("Loading model...")

nlp, pipes = load_model(
    negation=negation,
    family=family,
    hypothesis=hypothesis,
    reported_speech=reported_speech,
    custom_label=custom_label,
    custom_regex=custom_regex,
)

model_load_state.empty()

st.header("Enter a text to analyse:")
text = st.text_area(
    "Modify the following text and see the pipeline react :",
    DEFAULT_TEXT,
    height=330,
)

doc = nlp(text)

st.header("Visualisation")

st.markdown(
    "The pipeline extracts simple entities using a dictionnary of RegEx (see the "
    "[Export the pipeline section](#export-the-pipeline) for more information)."
)

colors = {
    "covid": "orange",
    "traitement": "#ff6363",
    "respiratoire": "#37b9fa",
    custom_label: "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
}
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

    d = dict(
        start=ent.start_char,
        end=ent.end_char,
        lexical_variant=ent.text,
        label=ent.label_,
    )

    if negation:
        d["negation"] = ent._.negation_

    if family:
        d["family"] = ent._.family_

    if hypothesis:
        d["hypothesis"] = ent._.hypothesis_

    if reported_speech:
        d["rspeech"] = ent._.reported_speech_

    data.append(d)

df = pd.DataFrame.from_records(data)

st.header("Entity qualification")

st.dataframe(df)

code = CODE.format(
    custom_label=custom_label,
    custom_regex=custom_regex,
    pipes="\n".join(pipes),
    text=f'"""\n{text}\n"""',
)

st.header("Export the pipeline")
st.markdown(
    "The code below recreates the pipeline. Copy and paste it "
    "in a Jupyter Notebook to interact with it."
)
with st.expander("Show the runnable code"):
    st.markdown(f"```python\n{code}\n```\n\nThis code runs as is.")


# st.header("JSON Doc")
# if st.button("Show JSON Doc"):
#     st.json(doc.to_json())

# st.header("JSON model meta")
# if st.button("Show JSON model meta"):
#     st.json(nlp.meta)
