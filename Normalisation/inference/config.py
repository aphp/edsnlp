import pandas as pd


######################
# ## GENERAL CONFIG ###
# #####################
umls_path = "/export/home/cse200093/scratch/BioMedics/data/umls/bio_str_SNOMEDCT_US.json"
labels_column_name = "CUI"
# Name of the column which contains the CUIs
synonyms_column_name = "STR"
# Name of the column which contains the synonyms
res_path = "/export/home/cse200093/Jacques_Bio/BioMedics/data/maladie_de_takayasu_norm/res.json"


####################
# ## CODER CONFIG ###
# ###################
column_name_to_normalize = "term"
# Name of the preceding column of interest. Default should be
# "terms_linked_to_measurement" to make the entire pipe work
coder_model_name_or_path = "/export/home/cse200093/scratch/word-embedding/coder_eds/model_967662.pth"
coder_tokenizer_name_or_path = "/export/home/cse200093/scratch/word-embedding/finetuning-camembert-2021-07-29"
coder_device = "cuda:0"
coder_save_umls_embeddings_dir = False
# set to False if you don't want to save
coder_save_umls_des_dir = False
# set to False if you don't want to save
coder_save_umls_labels_dir = False
# set to False if you don't want to save
coder_save_data_embeddings_dir = False
# set to False if you don't want to save
coder_normalize=True
coder_summary_method="CLS"
coder_tqdm_bar=True
coder_cased = True
coder_batch_size = 128
coder_stopwords = [
    "for",
    "assay",
    "by",
    "tests",
    "minute",
    "exam",
    "with",
    "human",
    "moyenne",
    "in",
    "to",
    "from",
    "analyse",
    "test",
    "level",
    "fluid",
    "laboratory",
    "determination",
    "examination",
    "releasing",
    "quantitative",
    "screening",
    "and",
    "exploration",
    "factor",
    "method",
    "analysis",
    "laboratoire",
    "specimen",
    "or",
    "typing",
    "of",
    "concentration",
    "measurement",
    "detection",
    "procedure",
    "identification",
    "numeration",
    "hour",
    "retired",
    "technique",
    "count",
]
coder_remove_stopwords_terms = False
coder_remove_special_characters_terms = False
coder_remove_stopwords_umls = True
coder_remove_special_characters_umls = True
