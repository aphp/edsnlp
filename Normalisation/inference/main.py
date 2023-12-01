import os
import typer
os.environ["OMP_NUM_THREADS"] = "16"
from text_preprocessor import TextPreprocessor
from get_normalization_with_coder import CoderNormalizer
from config import *
import pickle
from pathlib import Path

import pandas as pd

def coder_wrapper(df):
    # This wrapper is needed to preprocess terms
    # and in case the cells contains list of terms instead of one unique term
    df = df.reset_index(drop=True)
    text_preprocessor = TextPreprocessor(
        cased=coder_cased,
        stopwords=coder_stopwords
    )
    coder_normalizer = CoderNormalizer(
        model_name_or_path = coder_model_name_or_path,
        tokenizer_name_or_path = coder_tokenizer_name_or_path,
        device = coder_device
    )
    
    # Preprocess UMLS
    print("--- Preprocessing UMLS ---")
    umls_df = pd.read_json(umls_path)
    
    umls_df[synonyms_column_name] = umls_df[synonyms_column_name].apply(lambda term:
                                                                        text_preprocessor(
                                                                            text = term,
                                                                            remove_stopwords = coder_remove_stopwords_umls,
                                                                            remove_special_characters = coder_remove_special_characters_umls)
                                                                       )
    umls_df = (
        umls_df.loc[(~umls_df[synonyms_column_name].str.isnumeric()) & (umls_df[synonyms_column_name] != "")]
        .groupby([synonyms_column_name])
        .agg({labels_column_name: set, synonyms_column_name: "first"})
        .reset_index(drop=True)
    )
    coder_umls_des_list = umls_df[synonyms_column_name]
    coder_umls_labels_list = umls_df[labels_column_name]
    if coder_save_umls_des_dir:
        with open(coder_save_umls_des_dir, "wb") as f:
            pickle.dump(coder_umls_des_list, f)
    if coder_save_umls_labels_dir:
        with open(coder_save_umls_labels_dir, "wb") as f:
            pickle.dump(coder_umls_labels_list, f)
    
    # Preprocessing and inference on terms
    print("--- Preprocessing terms ---")
    if type(df[column_name_to_normalize].iloc[0]) == str:
        coder_data_list = df[column_name_to_normalize].apply(lambda term:
                                                            text_preprocessor(
                                                                text = term,
                                                                remove_stopwords = coder_remove_stopwords_terms,
                                                                remove_special_characters = coder_remove_special_characters_terms)
                                                            ).tolist()
        print("--- CODER inference ---")
        coder_res = coder_normalizer(
            umls_labels_list = coder_umls_labels_list,
            umls_des_list = coder_umls_des_list,
            data_list = coder_data_list,
            save_umls_embeddings_dir = coder_save_umls_embeddings_dir,
            save_data_embeddings_dir = coder_save_data_embeddings_dir,
            normalize = coder_normalize,
            summary_method = coder_summary_method,
            tqdm_bar = coder_tqdm_bar,
            coder_batch_size = coder_batch_size,
        )
        df[["label", "des"]] = pd.DataFrame(zip(*coder_res))
    else:
        exploded_term_df = pd.DataFrame({
            "id": df.index,
            column_name_to_normalize: df[column_name_to_normalize]
        }).explode(column_name_to_normalize).reset_index(drop=True)
        coder_data_list = exploded_term_df[column_name_to_normalize].apply(lambda term:
                                                                           text_preprocessor(
                                                                               text = term,
                                                                               remove_stopwords = coder_remove_stopwords_terms,
                                                                               remove_special_characters = coder_remove_special_characters_terms)
                                                                          ).tolist()
        print("--- CODER inference ---")
        coder_res = coder_normalizer(
            umls_labels_list = coder_umls_labels_list,
            umls_des_list = coder_umls_des_list,
            data_list = coder_data_list,
            save_umls_embeddings_dir = coder_save_umls_embeddings_dir,
            save_data_embeddings_dir = coder_save_data_embeddings_dir,
            normalize = coder_normalize,
            summary_method = coder_summary_method,
            tqdm_bar = coder_tqdm_bar,
            coder_batch_size = coder_batch_size,
        )
        exploded_term_df[["label", "des"]] = pd.DataFrame(zip(*coder_res))
        df = pd.merge(df.drop(columns=[column_name_to_normalize]), exploded_term_df, left_index = True, right_on = "id").drop(columns=["id"]).reset_index(drop=True)
    return df


def coder_inference_cli(
    input_dir: Path,
    output_dir: Path,
):
    df = pd.read_json(input_dir)
    df = coder_wrapper(df)
    if res_path:
        df.to_json(output_dir)

if __name__ == "__main__":
    typer.run(coder_inference_cli)
