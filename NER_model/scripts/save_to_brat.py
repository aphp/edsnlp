import pandas as pd 
import numpy as np
import spacy
from edsnlp.connectors.brat import BratConnector
import re
import srsly
import typer
from spacy.scorer import Scorer

from spacy.tokens import Doc
from spacy.training import Example
from spacy import util
from spacy.cli._util import Arg, Opt, import_code, setup_gpu
from spacy.cli.evaluate import (
    print_prf_per_type,
    print_textcats_auc_per_cat,
    render_parses,
)

import re
from pathlib import Path
from typing import Any, Dict, Optional
from tqdm import tqdm

import os
from spacy.tokens import DocBin
from thinc.api import fix_random_seed
from wasabi import Printer

from eds_medic.corpus_reader import Corpus




def evaluate_cli(
    model: str = Arg(..., help="Model name or path"),  # noqa: E501
    data_path: Path = Arg(..., help="Location of binary evaluation data in .spacy format", exists=True),  # noqa: E501
    output: Optional[Path] = Opt(None, "--output", "-o", help="Output JSON file for metrics", dir_okay=False),  # noqa: E501
    docbin: Optional[Path] = Opt(None, "--docbin", help="Output Doc Bin path", dir_okay=False),  # noqa: E501
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),  # noqa: E501
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),  # noqa: E501
    gold_preproc: bool = Opt(False, "--gold-preproc", "-G", help="Use gold preprocessing"),  # noqa: E501
    displacy_path: Optional[Path] = Opt(None, "--displacy-path", "-dp", help="Directory to output rendered parses as HTML", exists=True, file_okay=False),  # noqa: E501
    displacy_limit: int = Opt(25, "--displacy-limit", "-dl", help="Limit of parses to render as HTML"),  # noqa: E501
):
    
    save(model,
        ### A DECOMMENTER ###
        #data_path = '../data/attr2/test',
        #output_brat ='../data/attr2/pred',
        # data_path = "/export/home/"cse200093/Jacques_Bio/data_bio/brat_annotated_bio_val/test",
        # output_brat = "/export/home/cse200093/Jacques_Bio/data_bio/brat_annotated_bio_val/test_eds-medic",
        #data_path = '/export/home/cse200093/RV_Inter_conf/unnested_sosydiso_qualifiers_final/test_,
        #output_brat = '/export/home/cse200093/RV_Inter_conf/unnested_sosydiso_qualifiers_final/pred',
        #data_path = '/export/home/cse200093/RV_Inter_conf/unnested_final/test',
        #output_brat = '/export/home/cse200093/RV_Inter_conf/unnested_final/pred',
        data_path='/export/home/cse200093/Jacques_Bio/data_bio/super_pipe_get_stats_by_section_on_cim10/pred/syndrome_des_anti-phospholipides_init',
        output_brat='/export/home/cse200093/Jacques_Bio/data_bio/super_pipe_get_stats_by_section_on_cim10/pred/syndrome_des_anti-phospholipides_pred2',
        output=output,
        docbin=docbin,
        use_gpu=use_gpu,
        gold_preproc=gold_preproc,
        displacy_path=displacy_path,
        displacy_limit=displacy_limit,
        silent=False,
    )

    
def save(
    model: str,
    output_brat: str,
    data_path: Path,
    output: Optional[Path] = None,
    docbin: Optional[Path] = None,
    use_gpu: int = -1,
    gold_preproc: bool = False,
    displacy_path: Optional[Path] = None,
    displacy_limit: int = 25,
    silent: bool = True,
    spans_key: str = "sc",
):
    setup_gpu(use_gpu, silent)

    #brat = BratConnector(data_path, attributes = {"Disorders_type":"Disorders_type",'SOSY_type':'SOSY_type','Chemical_and_drugs_type':'Chemical_and_drugs_type',
     #                                        'Concept_type':'Concept_type','negation':'negation','hypothetique':'hypothetique', 'family':'family','Medical_Procedure_type':'Medical_Procedure_type','gender_type':'gender_type'})
    brat = BratConnector(data_path, attributes = {"Negation":"Negation","Family": "Family", "Temporality":"Temporality","Certainty":"Certainty","Action":"Action"})
    empty = spacy.blank("fr")
    df_gold = brat.brat2docs(empty)
    df_gold.sort(key=lambda doc: doc.text)


    
    
    print('-- Model running --')
    #model_path = '/export/home/cse200093/Pierre_Medic/NEURAL_BASED_NER/inference_model/model-best'
    df_txt = [doc.text for doc in df_gold]
    model = spacy.load(model)
    model.add_pipe('clean-entities')
    #caanot find clean entities... --> from edsmedic.... import clean_entites
    df_txt_pred = []
    for doc in tqdm(df_txt, desc="Processing documents"):
        doc = model(doc)
        doc._.trf_data = None
        df_txt_pred.append(doc) 

    for i in range(len(df_txt_pred)):
        df_txt_pred[i]._.note_id = df_gold[i]._.note_id
    
    
    #for doc in df_txt:
        #for ent in doc.ents:
            #if ent._.Action:
                #print(ent, ent._.Action)
    
    print('-- try saving --')
    
    print('path: ',output_brat)
    brat = BratConnector(output_brat, attributes =  {"Negation":"Negation","Family": "Family", "Temporality":"Temporality","Certainty":"Certainty","Action":"Action"})

    brat.docs2brat(df_txt_pred)
    
    print('-- saved -- ')

    
    

    
if __name__ == "__main__":
    typer.run(evaluate_cli)
