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

## TODO
- [x] REMGARDER LE SEUIL pour la positivité
- [x] regarder les patients en communs
- [x] regarder Hémoglobine et DFG
- [x] Finish fine tuning of CODER-EDS. Just execute `/export/home/cse200093/Jacques_Bio/normalisation/py_files/train_coder.sh` file up to 1M iterations (To know the number of iteration, just take a look at where the weigths of CODER-EDS are saved, i.e at `/export/home/cse200093/Jacques_Bio/data_bio/coder_output`. The files are saved with the number of iterations in their names.). Evaluate this model then with the files in `/export/home/cse200093/Jacques_Bio/normalisation/notebooks/coder` for example.
- [X] Requêter les médicaments en structuré !
- [X] Finir la normalisation des médicaments NER
- [ ] Cleaner le code et mettre sur GitHub
- [ ] Récupérer les figures
- [ ] Commencer à rédiger


```python
%reload_ext autoreload
%autoreload 2
%reload_ext jupyter_black
sc.cancelAllJobs()
```

```python
import os

os.environ["OMP_NUM_THREADS"] = "16"
```

```python
from edsteva import improve_performances

spark, sc, sql = improve_performances(
    to_add_conf=[
        ("spark.yarn.max.executor.failures", "10"),
        ("spark.executor.memory", "32g"),
        ("spark.driver.memory", "32g"),
        ("spark.driver.maxResultSize", "16g"),
        ("spark.default.parallelism", 160),
        ("spark.shuffle.service.enabled", "true"),
        ("spark.sql.shuffle.partitions", 160),
        ("spark.yarn.am.memory", "4g"),
        ("spark.yarn.max.executor.failures", 10),
        ("spark.dynamicAllocation.enabled", True),
        ("spark.dynamicAllocation.minExecutors", "20"),
        ("spark.dynamicAllocation.maxExecutors", "20"),
        ("spark.executor.cores", "8"),
    ]
)
```

```python
import pandas as pd
from os.path import isfile, isdir, join, basename
from os import listdir, mkdir
import spacy
from edsnlp.processing import pipe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn3, venn2
import altair as alt
from functools import reduce
from knowledge import TO_BE_MATCHED

import sys

BRAT_DIR = "/export/home/cse200093/scratch/BioMedics/data/CRH"
RES_DIR = "/export/home/cse200093/scratch/BioMedics/data/bio_results"
RES_DRUG_DIR = "/export/home/cse200093/scratch/BioMedics/data/drug_results"
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Only execute the following cells if you want to recreate the inference dataset (i.e dataset based on CIM10). PLEASE USE THE ENV `[2.4.3] K8s Py3 client` FOR THIS DATASET CREATION PART !
<!-- #endregion -->

### Functions

```python
### CELLS TO CREATE THE DATASET CONTAINING ALL TXT FILES WE WANT TO STUDY:
### ALL PATIENTS WITH ONE LINE AT LEAST IN:
# - i2b2_observation_cim10 with correct CIM10 according to `TO_BE_MATCHED`
# - i2b2_observation_doc
# - i2b2_observation_lab (OPTIONAL)

# SHOW DATASETS
sql("USE cse_200093_20210402")
sql("SHOW tables").show(10, False)


# Save txt function
def save_to_txt(path, txt):
    with open(path, "w") as f:
        print(txt, file=f)


def get_docs_df(cim10_list, min_len=1000):
    ### If we filter on `i2b2_observation_lab`
    # docs = sql("""SELECT doc.instance_num, doc.observation_blob, cim10.concept_cd FROM i2b2_observation_doc AS doc
    #               JOIN i2b2_observation_cim10 AS cim10 ON doc.encounter_num = cim10.encounter_num
    #               WHERE ((doc.concept_cd == 'CR:CRH-HOSPI' OR doc.concept_cd == 'CR:CRH-S')
    #               AND EXISTS (SELECT lab.encounter_num FROM i2b2_observation_lab AS lab
    #               WHERE lab.encounter_num = doc.encounter_num))""")

    ### If we don't filter on `i2b2_observation_lab`
    docs = sql(
        """SELECT doc.instance_num, doc.observation_blob, doc.encounter_num, doc.patient_num, visit.age_visit_in_years_num, visit.start_date, cim10.concept_cd FROM i2b2_observation_doc AS doc
                  JOIN i2b2_observation_cim10 AS cim10 ON doc.encounter_num = cim10.encounter_num JOIN i2b2_visit AS visit ON doc.encounter_num = visit.encounter_num
                  WHERE (doc.concept_cd == 'CR:CRH-HOSPI' OR doc.concept_cd == 'CR:CRH-S')
                  """
    )
    ### Filter on cim10_list and export to Pandas
    docs_df = docs.filter(docs.concept_cd.isin(cim10_list)).toPandas().dropna()
    ### Keep documents with some information at least
    docs_df = docs_df.loc[docs_df["observation_blob"].apply(len) > min_len].reset_index(
        drop=True
    )
    docs_df = (
        docs_df.groupby("observation_blob")
        .agg(
            {
                "instance_num": set,
                "encounter_num": "first",
                "patient_num": "first",
                "age_visit_in_years_num": "first",
                "start_date": "first",
                "observation_blob": "first",
            }
        )
        .reset_index(drop=True)
    )
    docs_df["instance_num"] = docs_df["instance_num"].apply(
        lambda instance_num: "_".join(list(instance_num))
    )
    return docs_df


def get_bio_df(summary_docs):
    bio = sql(
        """SELECT bio.instance_num AS bio_id, bio.concept_cd, bio.units_cd, bio.nval_num, bio.tval_char, bio.quantity_num, bio.confidence_num, bio.encounter_num, bio.patient_num, bio.start_date, concept.name_char 
        FROM i2b2_observation_lab AS bio JOIN i2b2_concept AS concept ON bio.concept_cd = concept.concept_cd"""
    )
    bio_dfs = {}
    for disease in summary_docs.disease.unique():
        unique_visit = summary_docs[summary_docs.disease == disease][
            ["encounter_num"]
        ].drop_duplicates()
        unique_visit = spark.createDataFrame(unique_visit).hint("broadcast")
        bio_df = bio.join(unique_visit, on="encounter_num").toPandas()
        bio_df["disease"] = disease
        bio_df["terms_linked_to_measurement"] = bio_df["name_char"].apply(
            _get_term_from_c_name
        )
        bio_df.loc[bio_df["units_cd"].isna(), "units_cd"] = "nounit"
        bio_df = bio_df[~((bio_df.nval_num.isna()) & (bio_df.tval_char.isna()))]
        display(bio_df)
        bio_dfs[disease] = bio_df

    return bio_dfs


def get_med_df(summary_docs):
    med = sql(
        """SELECT med.instance_num AS med_id, med.concept_cd, med.valueflag_cd, med.encounter_num, med.patient_num, med.start_date, concept.name_char 
        FROM i2b2_observation_med AS med JOIN i2b2_concept AS concept ON med.concept_cd = concept.concept_cd"""
    )
    med_dfs = {}
    for disease in summary_docs.disease.unique():
        unique_visit = summary_docs[summary_docs.disease == disease][
            ["encounter_num"]
        ].drop_duplicates()
        unique_visit = spark.createDataFrame(unique_visit).hint("broadcast")
        med_df = med.join(unique_visit, on="encounter_num").toPandas()
        med_df["disease"] = disease
        display(med_df)
        med_dfs[disease] = med_df

    return med_dfs


def _get_term_from_c_name(c_name):
    return c_name[c_name.index(":") + 1 :].split("_")[0].strip()
```

### Get Docs and Bio and Med

```python
# Get docs and save It for each disease
docs_all_diseases = []
for disease, disease_data in TO_BE_MATCHED.items():
    path_to_brat = join(BRAT_DIR, "raw", disease)
    if not os.path.exists(path_to_brat):
        mkdir(path_to_brat)
    docs_df = get_docs_df(["CIM10:" + cim10 for cim10 in disease_data["CIM10"]])
    docs_df.apply(lambda row: save_to_txt(join(path_to_brat, row["instance_num"] + ".txt"), row["observation_blob"]), axis=1)
    for file in os.listdir(path_to_brat):
        if file.endswith(".txt"):
            ann_file = os.path.join(path_to_brat, file[:-3] + "ann")
            open(ann_file, mode='a').close()
    print(disease + " processed and saved")
    docs_df["disease"] = disease
    docs_all_diseases.append(docs_df)
summary_df_docs = pd.concat(docs_all_diseases)
bio_from_structured_data = get_bio_df(summary_df_docs)
bio_from_structured_data = pd.concat(list(bio_from_structured_data.values()))
med_from_structured_data = get_med_df(summary_df_docs)
med_from_structured_data = pd.concat(list(med_from_structured_data.values()))
display(summary_df_docs)
display(bio_from_structured_data)
display(med_from_structured_data)
bio_from_structured_data.to_pickle(join(RES_DIR, "bio_from_structured_data.pkl"))
med_from_structured_data.to_pickle(join(RES_DRUG_DIR, "med_from_structured_data.pkl"))
summary_df_docs.to_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
```

```python
bio_from_structured_data["found"] = bio_from_structured_data["nval_num"].mask(
    bio_from_structured_data["nval_num"].isna(), bio_from_structured_data["tval_char"]
)
bio_from_structured_data["gold"] = (
    bio_from_structured_data["found"].astype(str) + " " + bio_from_structured_data["units_cd"]
)
bio_from_structured_data = bio_from_structured_data.groupby(
    ["disease", "encounter_num", "patient_num", "terms_linked_to_measurement"],
    as_index=False,
).agg({"name_char": list, "gold": list})
bio_from_structured_data.to_json(join(RES_DIR, "bio_from_structured_data.json"))
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Summary description of the data
<!-- #endregion -->

```python
import altair as alt

summary_df_docs = pd.read_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
bio_from_structured_data = pd.read_pickle(join(RES_DIR, "bio_from_structured_data.pkl"))
med_from_structured_data = pd.read_pickle(
    join(RES_DRUG_DIR, "med_from_structured_data.pkl")
)
```

## Number of docs/visit/patients

```python
summary_df_docs.groupby("disease").agg(
    {"instance_num": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

## Number of Bio/visit/patient

```python
bio_from_structured_data.groupby("disease").agg(
    {"bio_id": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

## Number of Med/visit/patient

```python
med_from_structured_data.groupby("disease").agg(
    {"med_id": "nunique", "encounter_num": "nunique", "patient_num": "nunique"}
)
```

## Age histogram

```python
summary_df_docs["round_age"] = (summary_df_docs["age_visit_in_years_num"] * 2).round(
    -1
) / 2
age_summary = summary_df_docs.groupby(
    ["disease", "age_visit_in_years_num"], as_index=False
).agg({"patient_num": "nunique"})
round_age_summary = summary_df_docs.groupby(
    ["disease", "round_age"], as_index=False
).agg({"patient_num": "nunique"})
total_patient = (
    summary_df_docs.groupby("disease", as_index=False)
    .agg({"patient_num": "nunique"})
    .rename(columns={"patient_num": "total_patient"})
)
age_summary = age_summary.merge(total_patient, on="disease")
age_summary["density"] = age_summary["patient_num"] / age_summary["total_patient"]
display(age_summary)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(round_age_summary).mark_bar(size=12, align="left").encode(
    alt.X("round_age:Q").title("Age at stay"),
    alt.Y("patient_num:Q").title("Number of patients"),
    alt.Row("disease:N"),
).resolve_scale(y="independent").properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(round_age_summary).mark_area(interpolate="step-after").encode(
    alt.X("round_age:Q").title("Age at stay"),
    alt.Y("patient_num:Q").title("Number of patients"),
    alt.Row("disease:N"),
).resolve_scale(y="independent").properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_area().encode(
    alt.X("age_visit_in_years_num:Q").title("Age at stay"),
    alt.Y("patient_num:Q").title("Number of patients"),
    alt.Row("disease:N"),
).resolve_scale(y="independent").properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_area(interpolate="basis").encode(
    alt.X("age_visit_in_years_num:Q").title("Age at stay"),
    alt.Y("density:Q").title("Density"),
    alt.Row("disease:N"),
).properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_bar().encode(
    alt.X("age_visit_in_years_num:Q"),
    alt.Y("density:Q"),
    alt.Row("disease:N"),
    color="disease:N",
).properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_area(opacity=0.4).encode(
    alt.X("age_visit_in_years_num:Q"), alt.Y("density:Q").stack(None), color="disease:N"
).properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(age_summary).mark_area().encode(
    alt.X("age_visit_in_years_num:Q"),
    alt.Y("density:Q").stack(True),
    color="disease:N",
).properties(height=200)
```

## Stay start histogramm

```python
summary_df_docs["month_date"] = (
    summary_df_docs["start_date"].dt.strftime("%Y-%m").astype("datetime64[ns]")
)
month_date_summary = summary_df_docs.groupby(
    ["disease", "month_date"], as_index=False
).agg({"encounter_num": "nunique"})
total_visit = (
    summary_df_docs.groupby("disease", as_index=False)
    .agg({"encounter_num": "nunique"})
    .rename(columns={"encounter_num": "total_visit"})
)
month_date_summary = month_date_summary.merge(total_visit, on="disease")
month_date_summary["density"] = (
    month_date_summary["encounter_num"] / month_date_summary["total_visit"]
)
display(month_date_summary)
```

```python
alt.data_transformers.disable_max_rows()
alt.Chart(month_date_summary).mark_bar(align="left").encode(
    alt.X("yearquarter(month_date):T")
    .title("Time (Year)")
    .axis(tickCount="year", labelAngle=0, grid=True, format="%Y"),
    alt.Y("sum(encounter_num):Q").title("Number of stays"),
    alt.Row("disease:N"),
).resolve_scale(y="independent").properties(height=200, width=600)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(month_date_summary).mark_area(interpolate="basis").encode(
    alt.X("month_date:T").title("Time (Year)"),
    alt.Y("density:Q").title("Density"),
    alt.Row("disease:N"),
).properties(height=200, width=600)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(month_date_summary).mark_bar().encode(
    alt.X("month_date:T").title("Time (Year)"),
    alt.Y("density:Q").title("Density"),
    alt.Row("disease:N"),
    color="disease:N",
).properties(height=200)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(month_date_summary).mark_area(opacity=0.4).encode(
    alt.X("month_date:T"), alt.Y("density:Q").stack(None), color="disease:N"
).properties(height=200, width=600)
```

```python
alt.data_transformers.disable_max_rows()

alt.Chart(month_date_summary).mark_area().encode(
    alt.X("month_date:T"),
    alt.Y("density:Q").stack(True),
    color="disease:N",
).properties(height=200)
```

# Please infer super_pipe on `BRAT_DIR` subfolders. Use the sbatch file in `/export/home/cse200093/Jacques_Bio/super_pipe/py_files/sbatch/main.sh`. res path should be in `RES_DIR`. NOW, PLEASE USE `jacques-spark` FOR THE NEXT CELLS.


## MED STRUCTURED

```python
med_from_structured_data = pd.read_pickle(
    join(RES_DRUG_DIR, "med_from_structured_data.pkl")
)
codes_to_keep = {"disease": [], "valueflag_cd": [], "med": []}
for disease, disease_data in TO_BE_MATCHED.items():
    for label, code_list in disease_data["ATC_codes"].items():
        for code in code_list:
            codes_to_keep["disease"].append(disease)
            codes_to_keep["valueflag_cd"].append(code)
            codes_to_keep["med"].append(label)
filtered_med = med_from_structured_data.merge(
    pd.DataFrame(codes_to_keep), on=["disease", "valueflag_cd"]
)
for disease in TO_BE_MATCHED.keys():
    path_to_res = join(RES_DRUG_DIR, disease)
    if not os.path.exists(path_to_res):
        mkdir(path_to_res)
    filtered_med[filtered_med.disease == disease].to_pickle(
        join(path_to_res, "filtered_med_from_structured_data.pkl")
    )
display(filtered_med)
filtered_med.to_pickle(join(RES_DRUG_DIR, "filtered_med_from_structured_data.pkl"))
```

## BIO STRUCTURED

```python
bio_from_structured_data = pd.read_pickle(join(RES_DIR, "bio_from_structured_data.pkl"))
codes_to_keep = {"disease": [], "concept_cd": [], "bio": []}
for disease, disease_data in TO_BE_MATCHED.items():
    for label, code_list in disease_data["ANABIO_codes"].items():
        for code in code_list:
            codes_to_keep["disease"].append(disease)
            codes_to_keep["concept_cd"].append(f"LAB:{code}")
            codes_to_keep["bio"].append(label)
filtered_bio = bio_from_structured_data.merge(
    pd.DataFrame(codes_to_keep), on=["disease", "concept_cd"]
)
for disease in TO_BE_MATCHED.keys():
    path_to_res = join(RES_DIR, disease)
    if not os.path.exists(path_to_res):
        mkdir(path_to_res)
    filtered_bio[filtered_bio.disease == disease].to_pickle(
        join(path_to_res, "filtered_bio_from_structured_data.pkl")
    )
display(filtered_bio)
filtered_bio.to_pickle(join(RES_DIR, "filtered_bio_from_structured_data.pkl"))
```

```python
bio_from_structured_data = pd.read_json(
    join(RES_DIR, "bio_from_structured_data.json"),
    dtype={"encounter_num": str, "patient_num": str},
).explode("label")
cuis_to_keep = {"disease": [], "label": [], "bio": []}
for disease, disease_data in TO_BE_MATCHED.items():
    for cui_dic in disease_data["CUI_per_section"].values():
        for cui_label, cui_list in cui_dic.items():
            for cui in cui_list:
                print(cui)
                cuis_to_keep["disease"].append(disease)
                cuis_to_keep["label"].append(cui)
                cuis_to_keep["bio"].append(cui_label)
filtered_bio_from_structured = bio_from_structured_data.merge(
    pd.DataFrame(cuis_to_keep), on=["disease", "label"]
)
for disease in TO_BE_MATCHED.keys():
    path_to_res = join(RES_DIR, disease)
    if not os.path.exists(path_to_res):
        mkdir(path_to_res)
    filtered_bio_from_structured[filtered_bio_from_structured.disease == disease].to_pickle(
        join(path_to_res, "filtered_bio_from_structured_data.pkl")
    )
display(filtered_bio_from_structured)
filtered_bio_from_structured.to_pickle(join(RES_DIR, "filtered_bio_from_structured_data.pkl"))
```

## MED NLP

```python
from tqdm import tqdm


# Check if we have to keep a match or not based on section and CUI
def keep_match(atc, atcs, atcs_to_keep):
    if atc not in atcs_to_keep:
        return None
    for drug, atc_list in atcs.items():
        if atc in atc_list:
            return drug
    return None


# List of df by disease for concatenation
res_part_filtered_list = []
res_part_df_list = []
for disease, disease_data in TO_BE_MATCHED.items():
    ### Load each res dataset to concat them in one unique df
    res_part_df = pd.read_pickle(join(RES_DRUG_DIR, disease, "norm_lev_match.pkl"))
    res_part_df["disease"] = disease
    res_part_df["source"] = res_part_df["source"] + ".ann"

    ### Filter ATC to keep
    codes_to_keep = {"label": [], "med": []}
    for label, code_list in disease_data["ATC_codes"].items():
        for code in code_list:
            codes_to_keep["label"].append(code)
            codes_to_keep["med"].append(label)
    res_part_filtered = (
        res_part_df.explode("label")
        .merge(pd.DataFrame(codes_to_keep), on="label")
        .drop_duplicates(
            subset=["term", "source", "span_converted", "norm_term", "disease"]
        )
    )

    ### Save for future concatenation
    res_part_filtered.to_pickle(join(RES_DRUG_DIR, disease, "res_final_filtered.pkl"))
    res_part_filtered_list.append(res_part_filtered)
res_filtered_df = pd.concat(res_part_filtered_list)
res_filtered_df.to_pickle(join(RES_DRUG_DIR, "res_final_filtered.pkl"))
display(res_filtered_df)
```

## BIO NLP WITHOUT SECTION

```python
from tqdm import tqdm


# Check if we have to keep a match or not based on section and CUI
def keep_match(cui, cui_per_section, cuis_to_keep):
    if cui not in cuis_to_keep:
        return None
    for bio, cui_list in cui_per_section["all"].items():
        if cui in cui_list:
            return bio
    return None


# List of df by disease for concatenation
res_part_filtered_list = []
res_part_df_list = []
for disease, disease_data in TO_BE_MATCHED.items():
    ### Load each res dataset to concat them in one unique df
    res_part_df = pd.read_json(join(RES_DIR, disease, "norm_coder_all.json"))
    res_part_df["disease"] = disease

    ### Filter CUIS to keep
    cuis_to_keep = [
        cui
        for cui_dic in disease_data["CUI_per_section"].values()
        for cui_list in cui_dic.values()
        for cui in cui_list
    ]
    res_part_filtered = []
    for source in tqdm(res_part_df["source"].unique()):
        for _, row in res_part_df.loc[res_part_df["source"] == source].iterrows():
            for cui in row["label"]:
                to_keep = keep_match(
                    cui,
                    disease_data["CUI_per_section"],
                    cuis_to_keep,
                )
                if to_keep:
                    row["bio"] = to_keep
                    res_part_filtered.append(row)

    ### Save for future concatenation
    res_part_df.to_pickle(join(RES_DIR, disease, "res_final.pkl"))
    res_part_df_list.append(res_part_df)
    pd.DataFrame(res_part_filtered).to_pickle(
        join(RES_DIR, disease, "res_final_filtered.pkl")
    )
    res_part_filtered_list += res_part_filtered
res_df = pd.concat(res_part_df_list)
res_filtered_df = pd.DataFrame(res_part_filtered_list)
res_df.to_pickle(join(RES_DIR, "res_final.pkl"))
res_filtered_df.to_pickle(join(RES_DIR, "res_final_filtered.pkl"))
display(res_df)
display(res_filtered_df)
```

## BIO NLP WITH SECTION

```python
from tqdm import tqdm

rule_based_section = False

if rule_based_section:
    # Load nlp pipe to detect sections
    nlp_sections = spacy.blank("eds")
    nlp_sections.add_pipe("eds.normalizer")
    nlp_sections.add_pipe("eds.sections")


# Check if two spans are overlapping for section detection
def is_overlapping(a, b):
    # Return true if a segment is overlapping b
    # else False
    return min(a[1], b[1]) > max(a[0], b[0])


# Check if we have to keep a match or not based on section and CUI
def keep_match(cui, span, txt_section_part_df, cui_per_section, cuis_to_keep):
    if cui not in cuis_to_keep:
        return None
    for section in cui_per_section.keys():
        if section == "all":
            for bio, cui_list in cui_per_section["all"].items():
                if cui in cui_list:
                    return bio
        elif section not in txt_section_part_df["label"].tolist():
            continue
        else:
            section_spans = (
                txt_section_part_df.loc[txt_section_part_df["label"] == section]
                .apply(lambda row: [row["start"], row["end"]], axis=1)
                .tolist()
            )
            for section_span in section_spans:
                if is_overlapping(span, section_span):
                    for bio, cui_list in cui_per_section[section].items():
                        if cui in cui_list:
                            return bio
            else:
                continue
    return None


# List of df by disease for concatenation
res_part_filtered_list = []
txt_sections_part_df_list = []
res_part_df_list = []
for disease, disease_data in TO_BE_MATCHED.items():
    ### Load each res dataset to concat them in one unique df
    res_part_df = pd.read_json(join(RES_DIR, disease, "norm_coder_all.json"))
    res_part_df["disease"] = disease

    if rule_based_section:
        ### Load txt files, detect sections and store it in df
        # Load txt files in DataFrame
        txt_files_part = [
            f
            for f in listdir(join(BRAT_DIR, "raw", disease))
            if isfile(join(BRAT_DIR, "raw", disease, f))
            if f.endswith(".txt")
        ]
        txt_list_part = []
        for txt_file in txt_files_part:
            with open(join(BRAT_DIR, "raw", disease, txt_file), "r") as file:
                text = file.read()
                txt_list_part.append([text, txt_file[:-3] + "ann"])
        txt_sections_part_df = pd.DataFrame(
            txt_list_part, columns=["note_text", "note_id"]
        )

        # Infer nlp pipe to detect sections
        txt_sections_part_df = pipe(
            note=txt_sections_part_df,
            nlp=nlp_sections,
            n_jobs=-2,
            additional_spans=["sections"],
        ).drop(columns=["span_type", "lexical_variant"])
    else:
        ### Load txt files, detect sections and store it in df
        # Load txt files in DataFrame
        txt_files_part = [
            f
            for f in listdir(join(BRAT_DIR, "pred", disease))
            if isfile(join(BRAT_DIR, "pred", disease, f))
            if f.endswith(".ann")
        ]
        txt_list_part = []
        for txt_file in txt_files_part:
            with open(join(BRAT_DIR, "pred", disease, txt_file), "r") as file:
                lines = file.readlines()
                start = 0
                section = "introduction"
                for line in lines:
                    if "SECTION" in line and not (
                        line.split("	")[1].split(" ")[0] == section
                    ):
                        end = int(line.split("	")[1].split(" ")[1])
                        txt_list_part.append([txt_file, section, start, end])
                        section = line.split("	")[1].split(" ")[0]
                        start = end
        txt_sections_part_df = pd.DataFrame(
            txt_list_part, columns=["note_id", "label", "start", "end"]
        )
    txt_sections_part_df["disease"] = disease

    ### Filter CUIS to keep
    sections_to_keep = list(disease_data["CUI_per_section"].keys())
    cuis_to_keep = [
        cui
        for cui_dic in disease_data["CUI_per_section"].values()
        for cui_list in cui_dic.values()
        for cui in cui_list
    ]
    print(cuis_to_keep)
    res_part_filtered = []
    for source in tqdm(res_part_df["source"].unique()):
        txt_sections_part_source_df = txt_sections_part_df.loc[
            (txt_sections_part_df["note_id"] == source)
            # & (txt_sections_part_df["label"].isin(sections_to_keep))
        ]
        for _, row in res_part_df.loc[res_part_df["source"] == source].iterrows():
            for cui in row["label"]:
                to_keep = keep_match(
                    cui,
                    row["span_converted"],
                    txt_sections_part_source_df,
                    disease_data["CUI_per_section"],
                    cuis_to_keep,
                )
                if to_keep:
                    row["bio"] = to_keep
                    res_part_filtered.append(row)

    ### Save for future concatenation
    res_part_df.to_pickle(join(RES_DIR, disease, "res_final.pkl"))
    res_part_df_list.append(res_part_df)
    pd.DataFrame(res_part_filtered).to_pickle(
        join(RES_DIR, disease, "res_final_filtered.pkl")
    )
    res_part_filtered_list += res_part_filtered
    txt_sections_part_df.to_pickle(join(RES_DIR, disease, "txt_sections_df.pkl"))
    txt_sections_part_df_list.append(txt_sections_part_df)

res_df = pd.concat(res_part_df_list)
res_filtered_df = pd.DataFrame(res_part_filtered_list)
txt_sections_df = pd.concat(txt_sections_part_df_list)
txt_sections_df.to_pickle(join(RES_DIR, "txt_sections_df.pkl"))
res_df.to_pickle(join(RES_DIR, "res_final.pkl"))
res_filtered_df.to_pickle(join(RES_DIR, "res_final_filtered.pkl"))
display(res_df)
display(res_filtered_df)
```

# Vizualize phenotype

```python
def prepare_structured_CODERE_label_df(disease):
    structured_filtered_res = pd.read_pickle(
        join(RES_DIR, disease, "structured_filtered_res.pkl")
    )
    summary_df_docs = pd.read_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    structured_filtered_res = structured_filtered_res.merge(
        summary_df_docs[["encounter_num", "patient_num"]],
        on=["encounter_num", "patient_num"],
        how="right",
    )
    structured_filtered_res = structured_filtered_res.explode("gold")
    structured_filtered_res["value"] = pd.to_numeric(
        structured_filtered_res["gold"].str.split(" ").str.get(0), errors="coerce"
    )
    structured_filtered_res["unit"] = (
        structured_filtered_res["gold"].str.split(" ").str.get(-1).str.lower()
    )
    structured_patient_group = None
    if len(TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()) > 0:
        for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys():
            structured_filtered_res[bio] = structured_filtered_res.bio == bio
            structured_filtered_res[f"{bio} positif"] = (
                structured_filtered_res.bio == bio
            ) & (structured_filtered_res.value >= 1.0)
        structured_patient_group = structured_filtered_res.groupby(
            "patient_num", as_index=False
        ).agg(
            {
                **{
                    bio: "sum"
                    for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()
                },
                **{
                    f"{bio} positif": "sum"
                    for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()
                },
            }
        )
        for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys():
            structured_patient_group[bio] = structured_patient_group[bio] >= 1
            structured_patient_group[f"{bio} positif"] = (
                structured_patient_group[f"{bio} positif"] >= 1
            )
    return structured_filtered_res, structured_patient_group
```

```python
def prepare_structured_df(disease):
    summary_filtered_res = pd.read_pickle(
        join(RES_DIR, disease, "filtered_bio_from_structured_data.pkl")
    )
    summary_df_docs = pd.read_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    summary_filtered_res = summary_filtered_res.merge(
        summary_df_docs[["encounter_num", "patient_num"]],
        on=["encounter_num", "patient_num"],
        how="right",
    )
    summary_filtered_res = summary_filtered_res.rename(
        columns={"nval_num": "value", "units_cd": "unit"}
    )
    summary_patient_group = None
    if len(TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()) > 0:
        for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys():
            summary_filtered_res[bio] = summary_filtered_res.bio == bio
            summary_filtered_res[f"{bio} positif"] = (
                summary_filtered_res.bio == bio
            ) & (
                (summary_filtered_res.value >= summary_filtered_res.confidence_num)
                | (summary_filtered_res.tval_char.str.contains("posi", case=False))
            )
        summary_patient_group = summary_filtered_res.groupby(
            "patient_num", as_index=False
        ).agg(
            {
                **{
                    bio: "sum"
                    for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()
                },
                **{
                    f"{bio} positif": "sum"
                    for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()
                },
            }
        )
        for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys():
            summary_patient_group[bio] = summary_patient_group[bio] >= 1
            summary_patient_group[f"{bio} positif"] = (
                summary_patient_group[f"{bio} positif"] >= 1
            )

    return summary_filtered_res, summary_patient_group
```

```python
def prepare_structured_med_df(disease):
    summary_filtered_res = pd.read_pickle(
        join(RES_DRUG_DIR, disease, "filtered_med_from_structured_data.pkl")
    )
    summary_df_docs = pd.read_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    summary_filtered_res = summary_filtered_res.merge(
        summary_df_docs[["encounter_num", "patient_num"]],
        on=["encounter_num", "patient_num"],
        how="right",
    )
    summary_patient_group = None
    for med in TO_BE_MATCHED[disease]["ATC_codes"].keys():
        summary_filtered_res[med] = summary_filtered_res.med == med
    summary_patient_group = summary_filtered_res.groupby(
        "patient_num", as_index=False
    ).agg(
        {
            **{med: "sum" for med in TO_BE_MATCHED[disease]["ATC_codes"].keys()},
        }
    )
    for med in TO_BE_MATCHED[disease]["ATC_codes"].keys():
        summary_patient_group[med] = summary_patient_group[med] >= 1

    return summary_filtered_res, summary_patient_group
```

```python
def prepare_nlp_med_df(disease):
    res_filtered_df = pd.read_pickle(
        join(RES_DRUG_DIR, disease, "res_final_filtered.pkl")
    )
    res_filtered_df["instance_num"] = (
        res_filtered_df.source.str.split(".").str.get(0).str.split("_").str.get(0)
    )
    summary_df_docs = pd.read_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    summary_df_docs["instance_num"] = summary_df_docs.instance_num.str.split("_")
    summary_df_docs = summary_df_docs.explode("instance_num")
    res_filtered_df = res_filtered_df.merge(
        summary_df_docs[["instance_num", "encounter_num", "patient_num"]],
        on="instance_num",
        how="right",
    )
    patient_group = None
    for med in TO_BE_MATCHED[disease]["ATC_codes"].keys():
        res_filtered_df[med] = res_filtered_df.med == med
    patient_group = res_filtered_df.groupby("patient_num", as_index=False).agg(
        {
            **{med: "sum" for med in TO_BE_MATCHED[disease]["ATC_codes"].keys()},
        }
    )
    for med in TO_BE_MATCHED[disease]["ATC_codes"].keys():
        patient_group[med] = patient_group[med] >= 1

    return res_filtered_df, patient_group
```

```python
def prepare_nlp_df(disease):
    res_filtered_df = pd.read_pickle(join(RES_DIR, disease, "res_final_filtered.pkl"))
    res_filtered_df["instance_num"] = (
        res_filtered_df.source.str.split(".").str.get(0).str.split("_").str.get(0)
    )
    summary_df_docs = pd.read_pickle(join(BRAT_DIR, "summary_df_docs.pkl"))
    summary_df_docs = summary_df_docs[summary_df_docs.disease == disease]
    summary_df_docs["instance_num"] = summary_df_docs.instance_num.str.split("_")
    summary_df_docs = summary_df_docs.explode("instance_num")
    res_filtered_df = res_filtered_df.merge(
        summary_df_docs[["instance_num", "encounter_num", "patient_num"]],
        on="instance_num",
        how="right",
    )
    res_filtered_df = res_filtered_df.explode("found")
    res_filtered_df["comparator"] = res_filtered_df["found"].str.split(" ").str.get(0)
    res_filtered_df["value"] = (
        res_filtered_df["found"].str.split(" ").str.get(1).astype(float)
    )
    res_filtered_df["unit"] = res_filtered_df["found"].str.split(" ").str.get(2)
    patient_group = None
    if len(TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()) > 0:
        for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys():
            res_filtered_df[bio] = res_filtered_df.bio == bio
            res_filtered_df[f"{bio} positif"] = (res_filtered_df.bio == bio) & (
                res_filtered_df.value >= 1.0
            )
        patient_group = res_filtered_df.groupby("patient_num", as_index=False).agg(
            {
                **{
                    bio: "sum"
                    for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()
                },
                **{
                    f"{bio} positif": "sum"
                    for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys()
                },
            }
        )
        for bio in TO_BE_MATCHED[disease]["CUI_per_section"]["all"].keys():
            patient_group[bio] = patient_group[bio] >= 1
            patient_group[f"{bio} positif"] = patient_group[f"{bio} positif"] >= 1

    return res_filtered_df, patient_group
```

```python
def plot_hist(unit_convert, possible_values, res_filtered_df, title: bool = False):
    alt.data_transformers.disable_max_rows()
    res_hists = []
    for bio, units in unit_convert.items():
        filtered_bio = res_filtered_df[["bio", "unit", "value"]][
            (res_filtered_df.bio == bio) & (res_filtered_df.unit.isin(units.keys()))
        ].copy()
        if not filtered_bio.empty:
            for unit, rate in units.items():
                filtered_bio["value"] = filtered_bio["value"].mask(
                    filtered_bio["unit"] == unit, filtered_bio["value"] * rate
                )
            outliers = filtered_bio[
                (filtered_bio["value"] > possible_values[bio])
                | (filtered_bio["value"] < 0)
            ].copy()
            outliers["Percentage"] = len(outliers) / len(filtered_bio)
            outliers["MaxValue"] = possible_values[bio]
            outliers["value"] = outliers["value"].mask(
                outliers["value"] > outliers["MaxValue"], outliers["MaxValue"]
            )
            outliers["value"] = outliers["value"].mask(outliers["value"] < 0, 0)
            filtered_bio = filtered_bio[
                (filtered_bio.value >= 0) & (filtered_bio.value <= possible_values[bio])
            ]
            res_density = (
                alt.Chart(filtered_bio)
                .transform_density(
                    "value",
                    counts=True,
                    extent=[0, possible_values[bio]],
                    as_=["value", "density"],
                )
                .mark_area()
                .encode(
                    alt.X("value:Q"),
                    alt.Y("density:Q"),
                    alt.Tooltip(["value:Q", "density:Q"]),
                )
            )
            res_box_plot = (
                alt.Chart(filtered_bio)
                .mark_boxplot()
                .encode(alt.X("value:Q").scale(domainMin=0))
            )
            res_outliers = (
                alt.Chart(outliers)
                .mark_bar(color="grey")
                .encode(
                    alt.X("value:Q"),
                    alt.Y("count()").title("Smoothed count"),
                    tooltip=[
                        alt.Tooltip(
                            "MaxValue:Q",
                            title="Upper bound",
                            format=",",
                        ),
                        alt.Tooltip(
                            "count():Q",
                            title="Frequency over the maximum",
                        ),
                        alt.Tooltip(
                            "Percentage:Q",
                            format=".2%",
                        ),
                    ],
                )
            )
            res_hist = (
                (res_density).properties(width=400, height=300) & res_box_plot
            ).resolve_scale(x="shared")
        else:
            res_hist = (
                alt.Chart(pd.DataFrame([]))
                .mark_text()
                .properties(width=400, height=300)
            )
        if title:
            res_hist = res_hist.properties(
                title=alt.TitleParams(text=bio, orient="top")
            )
        res_hists.append(res_hist)
    chart = reduce(
        lambda bar_chart_1, bar_chart_2: (bar_chart_1 | bar_chart_2)
        .resolve_scale(x="independent")
        .resolve_scale(y="independent"),
        res_hists,
    )
    return chart
```

```python
def plot_venn(patient_group, bio_venn, english_title, method):
    if len(bio_venn) == 2:
        subsets = (
            ((patient_group[bio_venn["A"]]) & ~(patient_group[bio_venn["B"]])).sum(),
            (~(patient_group[bio_venn["A"]]) & (patient_group[bio_venn["B"]])).sum(),
            ((patient_group[bio_venn["A"]]) & (patient_group[bio_venn["B"]])).sum(),
        )
        venn = venn2(subsets=subsets, set_labels=bio_venn.values())
    elif len(bio_venn) == 3:
        subsets = (
            (
                (patient_group[bio_venn["A"]])
                & ~(patient_group[bio_venn["B"]])
                & ~(patient_group[bio_venn["C"]])
            ).sum(),
            (
                ~(patient_group[bio_venn["A"]])
                & (patient_group[bio_venn["B"]])
                & ~(patient_group[bio_venn["C"]])
            ).sum(),
            (
                (patient_group[bio_venn["A"]])
                & (patient_group[bio_venn["B"]])
                & ~(patient_group[bio_venn["C"]])
            ).sum(),
            (
                ~(patient_group[bio_venn["A"]])
                & ~(patient_group[bio_venn["B"]])
                & (patient_group[bio_venn["C"]])
            ).sum(),
            (
                (patient_group[bio_venn["A"]])
                & ~(patient_group[bio_venn["B"]])
                & (patient_group[bio_venn["C"]])
            ).sum(),
            (
                ~(patient_group[bio_venn["A"]])
                & (patient_group[bio_venn["B"]])
                & (patient_group[bio_venn["C"]])
            ).sum(),
            (
                (patient_group[bio_venn["A"]])
                & (patient_group[bio_venn["B"]])
                & (patient_group[bio_venn["C"]])
            ).sum(),
        )
        venn = venn3(subsets=subsets, set_labels=bio_venn.values())

    total_patients = patient_group.patient_num.nunique()
    if len(bio_venn) == 3:
        total_pos = patient_group[
            patient_group[bio_venn["A"]]
            | patient_group[bio_venn["B"]]
            | patient_group[bio_venn["C"]]
        ].patient_num.nunique()
    elif len(bio_venn) == 2:
        total_pos = patient_group[
            patient_group[bio_venn["A"]] | patient_group[bio_venn["B"]]
        ].patient_num.nunique()
    for idx, subset in enumerate(venn.subset_labels):
        if subset:
            subset.set_text(
                f"{subset.get_text()}\n{int(subset.get_text())/total_patients*100:.1f}%"
            )
    plt.title(
        f"N = {total_patients} patients studied with a {english_title} \n Detected from {method} = {total_pos} ({total_pos/total_patients * 100:.1f} %)"
    )
    # plt.show()
```

```python
def plot_summary_med(nlp_patient_group, structured_patient_group, english_title):
    nlp_summary = pd.DataFrame(
        nlp_patient_group.sum().drop("patient_num"), columns=["Detected"]
    )
    nlp_summary["Total"] = len(nlp_patient_group)
    nlp_summary["Percentage"] = (
        nlp_summary["Detected"] / nlp_summary["Total"] * 100
    ).astype(float).round(2).astype(str) + " %"
    nlp_summary.columns = pd.MultiIndex.from_product(
        [
            ["NLP"],
            nlp_summary.columns,
        ]
    )
    structued_summary = pd.DataFrame(
        structured_patient_group.sum().drop("patient_num"), columns=["Detected"]
    )
    structued_summary["Total"] = len(structured_patient_group)
    structued_summary["Percentage"] = (
        (structued_summary["Detected"] / structued_summary["Total"] * 100)
        .astype(float)
        .round(2)
    ).astype(str) + " %"
    structued_summary.columns = pd.MultiIndex.from_product(
        [
            ["Structured Data"],
            structued_summary.columns,
        ]
    )
    nlp_structured_patient_group = (
        pd.concat([nlp_patient_group, structured_patient_group])
        .groupby("patient_num", as_index=False)
        .max()
    )
    nlp_structued_summary = pd.DataFrame(
        nlp_structured_patient_group.sum().drop("patient_num"), columns=["Detected"]
    )
    nlp_structued_summary["Total"] = len(nlp_structured_patient_group)
    nlp_structued_summary["Percentage"] = (
        (nlp_structued_summary["Detected"] / nlp_structued_summary["Total"] * 100)
        .astype(float)
        .round(2)
    ).astype(str) + " %"
    nlp_structued_summary.columns = pd.MultiIndex.from_product(
        [
            ["NLP + Structured Data"],
            nlp_structued_summary.columns,
        ]
    )
    return pd.concat(
        [structued_summary, nlp_summary, nlp_structued_summary], axis=1
    ).style.set_caption(english_title.capitalize())
```

```python
Biology_nlp_hist = []
Biology_structured_hist = []
Biology_nlp_structured_hist = []
unit_convert = {
    "Créatininémie": {"µmol_per_l": 1, "µmol/l": 1, "nounit": 1},
    "Hémoglobine": {"g_per_dl": 1, "g/dl": 1},
    "CRP": {"mg_per_l": 1, "µg_per_l": 0.001, "ui_per_l": 1, "nounit": 1, "mg/l": 1},
    "INR": {"nounit": 1},
    "DFG": {"ml_per_min": 1, "ml/min": 1, "nounit": 1, "mL/min/1,73m²": 1},
}
possible_values = {
    "Créatininémie": 1000,
    "Hémoglobine": 30,
    "CRP": 300,
    "INR": 10,
    "DFG": 200,
}
```

## syndrome_des_anti-phospholipides

```python
disease = "syndrome_des_anti-phospholipides"
english_title = "Antiphospholipid syndrome"
nlp_filtered_res, nlp_patient_group = prepare_nlp_df(disease)
structured_filtered_res, structured_patient_group = prepare_structured_df(disease)
_, nlp_patient_med_group = prepare_nlp_med_df(disease)
_, structured_patient_med_group = prepare_structured_med_df(disease)
structured_filtered_res["method"] = "structured_knowledge"
nlp_filtered_res["method"] = "nlp"
nlp_structured_filtered_res = pd.concat(
    [
        structured_filtered_res[
            ["encounter_num", "patient_num", "value", "unit", "bio", "method"]
        ],
        nlp_filtered_res[
            ["encounter_num", "patient_num", "value", "unit", "bio", "method"]
        ],
    ]
)
nlp_structured_patient_group = (
    pd.concat([nlp_patient_group, structured_patient_group])
    .groupby("patient_num", as_index=False)
    .max()
)
nlp_structured_patient_med_group = (
    pd.concat([nlp_patient_med_group, structured_patient_med_group])
    .groupby("patient_num", as_index=False)
    .max()
)
```

```python
med_venn = dict(A="Héparine", B="Anticoagulants oraux")
plot_venn(nlp_patient_med_group, med_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_nlp_med.jpeg")
plt.show()
plot_venn(
    structured_patient_med_group, med_venn, english_title, method="strctured data"
)
plt.savefig(f"figures/{disease}/venn_structured_med.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_med_group,
    med_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_nlp_structured_med.jpeg")
plt.show()
```

```python
plot_summary_med(nlp_patient_med_group, structured_patient_med_group, english_title)
```

```python
nlp_hists = plot_hist(unit_convert, possible_values, nlp_filtered_res, True).properties(
    title=english_title + " (NLP)"
)
strctured_hists = plot_hist(
    unit_convert, possible_values, structured_filtered_res, False
).properties(title=english_title + " (structured data)")
nlp_strctured_hists = plot_hist(
    unit_convert, possible_values, nlp_structured_filtered_res, False
).properties(title=english_title + " (NLP + structured data)")
chart = (
    (nlp_hists & strctured_hists & nlp_strctured_hists)
    .resolve_scale(x="independent")
    .resolve_scale(y="independent")
    .configure_title(anchor="middle", fontSize=20, orient="left")
)
if not os.path.exists(f"figures/{disease}"):
    os.makedirs(f"figures/{disease}")
chart.save(f"figures/{disease}/histogram.png")
chart.save(f"figures/{disease}/histogram.html")
# display(chart)
```

```python
Biology_nlp_hist.append(
    plot_hist(unit_convert, possible_values, nlp_filtered_res, True).properties(
        title=english_title + " (NLP)"
    )
)
Biology_structured_hist.append(
    plot_hist(unit_convert, possible_values, structured_filtered_res, True).properties(
        title=english_title + " (structured data)"
    )
)
Biology_nlp_structured_hist.append(
    plot_hist(
        unit_convert, possible_values, nlp_structured_filtered_res, True
    ).properties(title=english_title + " (NLP + structured data)")
)
```

```python
bio_venn = dict(
    A="Anti-cardiolipides", B="anti_B2GP1", C="anticoagulant_circulant_lupique"
)
plot_venn(nlp_patient_group, bio_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_nlp.jpeg")
plt.show()
plot_venn(structured_patient_group, bio_venn, english_title, method="strctured data")
plt.savefig(f"figures/{disease}/venn_structured.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_nlp_structured.jpeg")
plt.show()
```

```python
bio_venn = dict(
    A="Anti-cardiolipides positif",
    B="anti_B2GP1 positif",
    C="anticoagulant_circulant_lupique positif",
)
plot_venn(nlp_patient_group, bio_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_pos_nlp.jpeg")
plt.show()
plot_venn(structured_patient_group, bio_venn, english_title, method="strctured data")
plt.savefig(f"figures/{disease}/venn_pos_structured.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_pos_nlp_structured.jpeg")
plt.show()
```

## Lupus

FAN/AAN (C0587178), Anti-DNA Natif (C1262035)
Anti-Sm (C0201357) 

```python
disease = "lupus_erythemateux_dissemine"
english_title = "Lupus"
nlp_filtered_res, nlp_patient_group = prepare_nlp_df(disease)
structured_filtered_res, structured_patient_group = prepare_structured_df(disease)
_, nlp_patient_med_group = prepare_nlp_med_df(disease)
_, structured_patient_med_group = prepare_structured_med_df(disease)
structured_filtered_res["method"] = "structured_knowledge"
nlp_filtered_res["method"] = "nlp"
nlp_structured_filtered_res = pd.concat(
    [
        structured_filtered_res[
            ["encounter_num", "patient_num", "value", "unit", "bio", "method"]
        ],
        nlp_filtered_res[
            ["encounter_num", "patient_num", "value", "unit", "bio", "method"]
        ],
    ]
)
nlp_structured_patient_group = (
    pd.concat([nlp_patient_group, structured_patient_group])
    .groupby("patient_num", as_index=False)
    .max()
)
```

```python
plot_summary_med(nlp_patient_med_group, structured_patient_med_group, english_title)
```

```python
nlp_hists = plot_hist(unit_convert, possible_values, nlp_filtered_res, True).properties(
    title=english_title + " (NLP)"
)
strctured_hists = plot_hist(
    unit_convert, possible_values, structured_filtered_res, False
).properties(title=english_title + " (structured data)")
nlp_strctured_hists = plot_hist(
    unit_convert, possible_values, nlp_structured_filtered_res, False
).properties(title=english_title + " (NLP + structured data)")
chart = (
    (nlp_hists & strctured_hists & nlp_strctured_hists)
    .resolve_scale(x="independent")
    .resolve_scale(y="independent")
    .configure_title(anchor="middle", fontSize=20, orient="left")
)
if not os.path.exists(f"figures/{disease}"):
    os.makedirs(f"figures/{disease}")
chart.save(f"figures/{disease}/histogram.png")
chart.save(f"figures/{disease}/histogram.html")
# display(chart)
```

```python
Biology_nlp_hist.append(
    plot_hist(unit_convert, possible_values, nlp_filtered_res).properties(
        title=english_title + " (NLP)"
    )
)
Biology_structured_hist.append(
    plot_hist(unit_convert, possible_values, structured_filtered_res).properties(
        title=english_title + " (structured data)"
    )
)
Biology_nlp_structured_hist.append(
    plot_hist(unit_convert, possible_values, nlp_structured_filtered_res).properties(
        title=english_title + " (NLP + structured data)"
    )
)
```

```python
bio_venn = dict(A="Facteur anti-nucléaire", B="Anti-DNA natif", C="Anti-Sm")
plot_venn(nlp_patient_group, bio_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_nlp.jpeg")
plt.show()
plot_venn(structured_patient_group, bio_venn, english_title, method="strctured data")
plt.savefig(f"figures/{disease}/venn_structured.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_nlp_structured.jpeg")
plt.show()
```

```python
bio_venn = dict(
    A="Facteur anti-nucléaire positif", B="Anti-DNA natif positif", C="Anti-Sm positif"
)
plot_venn(nlp_patient_group, bio_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_pos_nlp.jpeg")
plt.show()
plot_venn(structured_patient_group, bio_venn, english_title, method="strctured data")
plt.savefig(f"figures/{disease}/venn_pos_structured.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_pos_nlp_structured.jpeg")
plt.show()
```

```python
bio_venn = dict(A="Facteur anti-nucléaire", B="Anti-DNA natif")
plot_venn(nlp_patient_group, bio_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_2_nlp.jpeg")
plt.show()
plot_venn(structured_patient_group, bio_venn, english_title, method="strctured data")
plt.savefig(f"figures/{disease}/venn_2_structured.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_2_nlp_structured.jpeg")
plt.show()
```

```python
bio_venn = dict(A="Facteur anti-nucléaire positif", B="Anti-DNA natif positif")
plot_venn(nlp_patient_group, bio_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_pos_2_nlp.jpeg")
plt.show()
plot_venn(structured_patient_group, bio_venn, english_title, method="strctured data")
plt.savefig(f"figures/{disease}/venn_pos_2_structured.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_pos_2_nlp_structured.jpeg")
plt.show()
```

## Sclérodermie systémique

```python
disease = "sclerodermie_systemique"
english_title = "systemic sclerosis"
nlp_filtered_res, nlp_patient_group = prepare_nlp_df(disease)
structured_filtered_res, structured_patient_group = prepare_structured_df(disease)
_, nlp_patient_med_group = prepare_nlp_med_df(disease)
_, structured_patient_med_group = prepare_structured_med_df(disease)
structured_filtered_res["method"] = "structured_knowledge"
nlp_filtered_res["method"] = "nlp"
nlp_structured_filtered_res = pd.concat(
    [
        structured_filtered_res[
            ["encounter_num", "patient_num", "value", "unit", "bio", "method"]
        ],
        nlp_filtered_res[
            ["encounter_num", "patient_num", "value", "unit", "bio", "method"]
        ],
    ]
)
nlp_structured_patient_group = (
    pd.concat([nlp_patient_group, structured_patient_group])
    .groupby("patient_num", as_index=False)
    .max()
)
```

```python
plot_summary_med(nlp_patient_med_group, structured_patient_med_group, english_title)
```

```python
nlp_hists = plot_hist(unit_convert, possible_values, nlp_filtered_res, True).properties(
    title=english_title + " (NLP)"
)
strctured_hists = plot_hist(
    unit_convert, possible_values, structured_filtered_res, False
).properties(title=english_title + " (structured data)")
nlp_strctured_hists = plot_hist(
    unit_convert, possible_values, nlp_structured_filtered_res, False
).properties(title=english_title + " (NLP + structured data)")
chart = (
    (nlp_hists & strctured_hists & nlp_strctured_hists)
    .resolve_scale(x="independent")
    .resolve_scale(y="independent")
    .configure_title(anchor="middle", fontSize=20, orient="left")
)
if not os.path.exists(f"figures/{disease}"):
    os.makedirs(f"figures/{disease}")
chart.save(f"figures/{disease}/histogram.png")
chart.save(f"figures/{disease}/histogram.html")
# display(chart)
```

```python
Biology_nlp_hist.append(
    plot_hist(unit_convert, possible_values, nlp_filtered_res).properties(
        title=english_title + " (NLP)"
    )
)
Biology_structured_hist.append(
    plot_hist(unit_convert, possible_values, structured_filtered_res).properties(
        title=english_title + " (structured data)"
    )
)
Biology_nlp_structured_hist.append(
    plot_hist(unit_convert, possible_values, nlp_structured_filtered_res).properties(
        title=english_title + " (NLP + structured data)"
    )
)
```

```python
bio_venn = dict(A="Anti-RNA pol 3", B="Anti-SCL 70")
plot_venn(nlp_patient_group, bio_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_nlp.jpeg")
plt.show()
plot_venn(structured_patient_group, bio_venn, english_title, method="strctured data")
plt.savefig(f"figures/{disease}/venn_structured.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_nlp_structured.jpeg")
plt.show()
```

```python
bio_venn = dict(A="Anti-RNA pol 3 positif", B="Anti-SCL 70 positif")
plot_venn(nlp_patient_group, bio_venn, english_title, method="discharge summaries")
plt.savefig(f"figures/{disease}/venn_pos_nlp.jpeg")
plt.show()
plot_venn(structured_patient_group, bio_venn, english_title, method="strctured data")
plt.savefig(f"figures/{disease}/venn_pos_structured.jpeg")
plt.show()
plot_venn(
    nlp_structured_patient_group,
    bio_venn,
    english_title,
    method="discharge summaries and structured data",
)
plt.savefig(f"figures/{disease}/venn_pos_nlp_structured.jpeg")
plt.show()
```

### Maladie de Takayasu

```python
disease = "maladie_de_takayasu"
english_title = "Takayasu´s disease"
nlp_filtered_res, _ = prepare_nlp_df(disease)
structured_filtered_res, _ = prepare_structured_df(disease)
_, nlp_patient_med_group = prepare_nlp_med_df(disease)
_, structured_patient_med_group = prepare_structured_med_df(disease)
structured_filtered_res["method"] = "structured_knowledge"
nlp_filtered_res["method"] = "nlp"
nlp_structured_filtered_res = pd.concat(
    [
        structured_filtered_res[
            ["encounter_num", "patient_num", "value", "unit", "bio", "method"]
        ],
        nlp_filtered_res[
            ["encounter_num", "patient_num", "value", "unit", "bio", "method"]
        ],
    ]
)
```

```python
plot_summary_med(nlp_patient_med_group, structured_patient_med_group, english_title)
```

```python
nlp_hists = plot_hist(unit_convert, possible_values, nlp_filtered_res, True).properties(
    title=english_title + " (NLP)"
)
strctured_hists = plot_hist(
    unit_convert, possible_values, structured_filtered_res, False
).properties(title=english_title + " (structured data)")
nlp_strctured_hists = plot_hist(
    unit_convert, possible_values, nlp_structured_filtered_res, False
).properties(title=english_title + " (NLP + structured data)")
chart = (
    (nlp_hists & strctured_hists & nlp_strctured_hists)
    .resolve_scale(x="independent")
    .resolve_scale(y="independent")
    .configure_title(anchor="middle", fontSize=20, orient="left")
)
if not os.path.exists(f"figures/{disease}"):
    os.makedirs(f"figures/{disease}")
chart.save(f"figures/{disease}/histogram.png")
chart.save(f"figures/{disease}/histogram.html")
# display(chart)
```

```python
Biology_nlp_hist.append(
    plot_hist(unit_convert, possible_values, nlp_filtered_res).properties(
        title=english_title + " (NLP)"
    )
)
Biology_structured_hist.append(
    plot_hist(unit_convert, possible_values, structured_filtered_res).properties(
        title=english_title + " (structured data)"
    )
)
Biology_nlp_structured_hist.append(
    plot_hist(unit_convert, possible_values, nlp_structured_filtered_res).properties(
        title=english_title + " (NLP + structured data)"
    )
)
```

```python
chart = reduce(
    lambda bar_chart_1, bar_chart_2: (bar_chart_1 & bar_chart_2)
    .resolve_scale(x="independent")
    .resolve_scale(y="independent"),
    Biology_nlp_hist,
).configure_title(orient="left", anchor="middle", fontSize=20)
chart.save("figures/histogram_nlp.png")
chart.save("figures/histogram_nlp.html")
# display(chart)
```

```python
chart = reduce(
    lambda bar_chart_1, bar_chart_2: (bar_chart_1 & bar_chart_2)
    .resolve_scale(x="independent")
    .resolve_scale(y="independent"),
    Biology_structured_hist,
).configure_title(orient="left", anchor="middle", fontSize=20)
chart.save("figures/histogram_structured.png")
chart.save("figures/histogram_structured.html")
# display(chart)
```

```python
chart = reduce(
    lambda bar_chart_1, bar_chart_2: (bar_chart_1 & bar_chart_2)
    .resolve_scale(x="independent")
    .resolve_scale(y="independent"),
    Biology_nlp_structured_hist,
).configure_title(orient="left", anchor="middle", fontSize=20)
chart.save("figures/histogram_nlp_structured.png")
chart.save("figures/histogram_nlp_structured.html")
# display(chart)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
res_filtered_df[res_filtered_df.bio == "DFG"].unit.value_counts()
```

```python
"Anti-B2GP1" =[
    "E9627",
    "A7854",
    "H3772",
    "E5157",
    "I2042",
    "I1882",
    "X9708",
    "H6269",
    "X5898",
    "X2761",
    "A7855",
    "H9650",
    "E9626",
    "I2043",
    "I1883",
    "X9707",
    "H6270",
    "X5899",
    "I5970",
    "H5543",
    "H6271",
    "J9678",
    "J9704",
    "J9705",
    "K8345",
]
```

```python
structured_filtered_res.explode("name_char").name_char.str.split(":").str.get(0).unique()
```

```python
structured_res = pd.read_json(
    join(BRAT_DIR, "summary_df_bio.json"),
    dtype={"encounter_num": str, "patient_num": str},
).explode("label")
```

```python
structured_res = structured_res.explode("name_char")
structured_res[structured_res.name_char.str.contains("F8160")]
```
