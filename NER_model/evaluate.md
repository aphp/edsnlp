---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: BioMedics_client
    language: python
    name: biomedics_client
---

```python
%reload_ext autoreload
%autoreload 2
%reload_ext jupyter_black
```

```python
import spacy
import pandas as pd
from edsnlp.connectors.brat import BratConnector
from edsnlp.evaluate import evaluate_test, evaluate
```

# Expe Data Size

```python
GOLD_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/test"

loader = spacy.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)

scores = []
for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62]:
    PRED_PATH = f"/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/expe_data_size/pred_{i}"
    loader = spacy.blank("eds")
    brat = BratConnector(PRED_PATH)
    pred_docs = brat.brat2docs(loader)
    score = pd.DataFrame(
        evaluate_test(
            gold_docs,
            pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=5,
        )
    ).T.sort_index()
    score[["n_docs"]] = i
    scores.append(score)
```

```python
import altair as alt
from functools import reduce

alt.data_transformers.disable_max_rows()
result = (
    pd.concat(scores)[["n_docs", "Precision", "Recall", "F1"]]
    .dropna()
    .reset_index()
    .rename(columns={"index": "label"})
    .melt(
        id_vars=["n_docs", "label"],
        value_vars=["Precision", "Recall", "F1"],
        var_name="metric",
        value_name="summary",
    )
)
result["mean"] = result["summary"].str.split().str.get(0)
result["lower"] = (
    result["summary"].str.split().str.get(1).str.split("-").str.get(0).str.slice(1)
)
result["upper"] = (
    result["summary"].str.split().str.get(1).str.split("-").str.get(1).str.slice(0, -1)
)
result = result[
    result.label.isin(
        [
            "Overall",
            "DISO",
            "Constantes",
            "BIO_comp",
            "Chemical_and_drugs",
            "dosage",
            "BIO",
            "strength",
            "form",
            "SECTION_antecedent",
            "SECTION_motif",
            "SECTION_histoire",
            "SECTION_examen_clinique",
            "SECTION_examen_complementaire",
            "SECTION_mode_de_vie",
            "SECTION_traitement_entree",
            "SECTION_antecedent_familiaux",
            "SECTION_traitement_sortie",
            "SECTION_conclusion",
        ]
    )
]
label_dropdown = alt.binding_select(options=list(result.label.unique()), name="Label ")
label_selection = alt.selection_point(
    fields=["label"], bind=label_dropdown, value="Overall"
)

metric_dropdown = alt.binding_select(
    options=list(result.metric.unique()), name="Metric "
)
metric_selection = alt.selection_point(
    fields=["metric"], bind=metric_dropdown, value="F1"
)

line = (
    alt.Chart(result)
    .mark_line(point=True)
    .encode(
        x="n_docs:O",
        y=alt.Y(f"mean:Q").scale(zero=False, domain=[60, 100]),
    )
)

band = (
    alt.Chart(result)
    .mark_area(opacity=0.5)
    .encode(
        x="n_docs:O",
        y=alt.Y(f"upper:Q").title(""),
        y2=alt.Y2(f"lower:Q").title(""),
    )
)

chart = line + band
chart = (
    chart.add_params(metric_selection)
    .transform_filter(metric_selection)
    .add_params(label_selection)
    .transform_filter(label_selection)
    .properties(width=600)
)

display(chart)
display(result)
chart.save("metrics_by_n_docs.html")
```

# Expe Section

```python
from os.path import isfile, isdir, join, basename
import edsnlp
import spacy
from edsnlp.evaluate import compute_scores

nlp = spacy.blank("eds")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sections")

GOLD_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/test"

loader = spacy.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)

ML_PRED_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/expe_lang_model/pred_model_eds_finetune"

brat = BratConnector(ML_PRED_PATH)
ents_ml_pred = brat.brat2docs(loader)

mapping = {
    "antécédents": "SECTION_antecedent",
    "motif": "SECTION_motif",
    "histoire de la maladie": "SECTION_histoire",
    "examens": "SECTION_examen_clinique",
    "examens complémentaires": "SECTION_examen_complementaire",
    "habitus": "SECTION_mode_de_vie",
    "traitements entrée": "SECTION_traitement_entree",
    "antécédents familiaux": "SECTION_antecedent_familiaux",
    "traitements sortie": "SECTION_traitement_sortie",
    "conclusion": "SECTION_conclusion",
}
rule_pred_docs = []
for doc in gold_docs:
    rule_pred_doc = nlp(doc.text)
    rule_pred_doc._.note_id = doc._.note_id
    del rule_pred_doc.spans["sections"]
    rule_pred_docs.append(rule_pred_doc)
ents_rule_pred = []
for doc in rule_pred_docs:
    annotation = [doc._.note_id]
    for label, ents in doc.spans.items():
        for ent in ents:
            if ent.label_ in mapping.keys():
                annotation.append(
                    [ent.text, mapping[ent.label_], ent.start_char, ent.end_char]
                )
    ents_rule_pred.append(annotation)


def get_annotation(docs):
    full_annot = []
    for doc in docs:
        annotation = [doc._.note_id]
        for label, ents in doc.spans.items():
            for ent in ents:
                if label in mapping.values():
                    annotation.append([ent.text, label, ent.start_char, ent.end_char])
        full_annot.append(annotation)
    return full_annot


ents_gold, ents_ml_pred = (
    get_annotation(gold_docs),
    get_annotation(ents_ml_pred),
)
ents_gold.sort(key=lambda l: l[0])
ents_ml_pred.sort(key=lambda l: l[0])
ents_rule_pred.sort(key=lambda l: l[0])

scores_rule = (
    pd.DataFrame(
        compute_scores(
            ents_gold=ents_gold,
            ents_pred=ents_rule_pred,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
        )
    )
    .T.sort_index()[["N_entity", "Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores_rule.columns = pd.MultiIndex.from_product(
    [["Rule-Based"], ["N_entity", "Precision", "Recall", "F1"]]
)

scores_ml = (
    pd.DataFrame(
        compute_scores(
            ents_gold=ents_gold,
            ents_pred=ents_ml_pred,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
        )
    )
    .T.sort_index()[["Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
        ]
    )
)
scores_ml.columns = pd.MultiIndex.from_product(
    [["ML (NER)"], ["Precision", "Recall", "F1"]]
)
result = scores_rule.merge(scores_ml, left_index=True, right_index=True)
result
```

```python
import numpy as np


def highlight_max(row):
    Precision_max = (
        row[:, "Precision"].str.split(" ").str.get(0).astype(float)
        == row[:, "Precision"].str.split(" ").str.get(0).astype(float).max()
    )
    Recall_max = (
        row[:, "Recall"].str.split(" ").str.get(0).astype(float)
        == row[:, "Recall"].str.split(" ").str.get(0).astype(float).max()
    )
    F1_max = (
        row[:, "F1"].str.split(" ").str.get(0).astype(float)
        == row[:, "F1"].str.split(" ").str.get(0).astype(float).max()
    )
    s_max = [False]
    for i in range(len(F1_max)):
        s_max.append(Precision_max[i])
        s_max.append(Recall_max[i])
        s_max.append(F1_max[i])
    return ["font-weight: bold" if cell else "" for cell in s_max]


def remove_confidence(row):
    return row[:, :].str.split(" ").str.get(0)


result.apply(remove_confidence, axis=1).style.apply(highlight_max, axis=1)
```

# Expe lang models

```python
import spacy

GOLD_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/test"

loader = spacy.blank("eds")
brat = BratConnector(GOLD_PATH)
gold_docs = brat.brat2docs(loader)

CAM_BASE_PRED_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/expe_lang_model/pred_model_camembert_base"

brat = BratConnector(CAM_BASE_PRED_PATH)
cam_base_pred_docs = brat.brat2docs(loader)

CAM_BIO_PRED_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/expe_lang_model/pred_model_camembert_bio"

brat = BratConnector(CAM_BIO_PRED_PATH)
cam_bio_pred_docs = brat.brat2docs(loader)

DR_BERT_PRED_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/expe_lang_model/pred_model_DrBert"

brat = BratConnector(DR_BERT_PRED_PATH)
DrBert_pred_docs = brat.brat2docs(loader)

EDS_FINE_PRED_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/expe_lang_model/pred_model_eds_finetune"

brat = BratConnector(EDS_FINE_PRED_PATH)
eds_finetune_pred_docs = brat.brat2docs(loader)

EDS_SCRATCH_PRED_PATH = "/export/home/cse200093/scratch/BioMedics/NER_model/data/NLP_diabeto/expe_lang_model/pred_model_eds_scratch"

brat = BratConnector(EDS_SCRATCH_PRED_PATH)
eds_scratch_pred_docs = brat.brat2docs(loader)
```

```python
scores_cam_base = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            cam_base_pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
        )
    )
    .T.rename(
        index={
            "DISO": "Diso",
            "Chemical_and_drugs": "Drugs",
            "dosage": "Drugs_Dosage",
            "form": "Drugs_Form",
            "strength": "Drugs_Strength",
            "Overall": "overall",
        }
    )
    .sort_index()[["N_entity", "Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
            "route",
            "SECTION_traitement",
            "SECTION_evolution",
            "BIO_milieu",
        ]
    )
)
scores_cam_base.columns = pd.MultiIndex.from_product(
    [["CamemBert-Base"], ["N_entity", "Precision", "Recall", "F1"]]
)

scores_cam_bio = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            cam_bio_pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
        )
    )
    .T.rename(
        index={
            "DISO": "Diso",
            "Chemical_and_drugs": "Drugs",
            "dosage": "Drugs_Dosage",
            "form": "Drugs_Form",
            "strength": "Drugs_Strength",
            "Overall": "overall",
        }
    )
    .sort_index()[["Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
            "route",
            "SECTION_traitement",
            "SECTION_evolution",
            "BIO_milieu",
        ]
    )
)
scores_cam_bio.columns = pd.MultiIndex.from_product(
    [["CamemBert-Bio"], ["Precision", "Recall", "F1"]]
)

scores_DrBert = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            DrBert_pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
        )
    )
    .T.rename(
        index={
            "DISO": "Diso",
            "Chemical_and_drugs": "Drugs",
            "dosage": "Drugs_Dosage",
            "form": "Drugs_Form",
            "strength": "Drugs_Strength",
            "Overall": "overall",
        }
    )
    .sort_index()[["Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
            "route",
            "SECTION_traitement",
            "SECTION_evolution",
            "BIO_milieu",
        ]
    )
)
scores_DrBert.columns = pd.MultiIndex.from_product(
    [["DrBert"], ["Precision", "Recall", "F1"]]
)

scores_eds_finetune = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            eds_finetune_pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
        )
    )
    .T.rename(
        index={
            "DISO": "Diso",
            "Chemical_and_drugs": "Drugs",
            "dosage": "Drugs_Dosage",
            "form": "Drugs_Form",
            "strength": "Drugs_Strength",
            "Overall": "overall",
        }
    )
    .sort_index()[["Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
            "route",
            "SECTION_traitement",
            "SECTION_evolution",
            "BIO_milieu",
        ]
    )
)
scores_eds_finetune.columns = pd.MultiIndex.from_product(
    [["CamemBert-EDS Finetuned"], ["Precision", "Recall", "F1"]]
)

scores_eds_scratch = (
    pd.DataFrame(
        evaluate_test(
            gold_docs,
            eds_scratch_pred_docs,
            boostrap_level="doc",
            exact=True,
            n_draw=5000,
            alpha=0.05,
            digits=2,
        )
    )
    .T.rename(
        index={
            "DISO": "Diso",
            "Chemical_and_drugs": "Drugs",
            "dosage": "Drugs_Dosage",
            "form": "Drugs_Form",
            "strength": "Drugs_Strength",
            "Overall": "overall",
        }
    )
    .sort_index()[["Precision", "Recall", "F1"]]
    .drop(
        index=[
            "ents_per_type",
            "route",
            "SECTION_traitement",
            "SECTION_evolution",
            "BIO_milieu",
        ]
    )
)
scores_eds_scratch.columns = pd.MultiIndex.from_product(
    [["CamemBert-EDS Scratch"], ["Precision", "Recall", "F1"]]
)

result = (
    scores_cam_base.merge(scores_cam_bio, left_index=True, right_index=True)
    .merge(scores_DrBert, left_index=True, right_index=True)
    .merge(scores_eds_finetune, left_index=True, right_index=True)
    .merge(scores_eds_scratch, left_index=True, right_index=True)
)
```

```python
import numpy as np


def highlight_max(row):
    Precision_max = (
        row[:, "Precision"].str.split(" ").str.get(0).astype(float)
        == row[:, "Precision"].str.split(" ").str.get(0).astype(float).max()
    )
    Recall_max = (
        row[:, "Recall"].str.split(" ").str.get(0).astype(float)
        == row[:, "Recall"].str.split(" ").str.get(0).astype(float).max()
    )
    F1_max = (
        row[:, "F1"].str.split(" ").str.get(0).astype(float)
        == row[:, "F1"].str.split(" ").str.get(0).astype(float).max()
    )
    s_max = [False]
    for i in range(len(F1_max)):
        s_max.append(Precision_max[i])
        s_max.append(Recall_max[i])
        s_max.append(F1_max[i])
    return ["font-weight: bold" if cell else "" for cell in s_max]


def remove_confidence(row):
    return row[:, :].str.split(" ").str.get(0)


result.apply(remove_confidence, axis=1).style.apply(highlight_max, axis=1)
```

```python

```
