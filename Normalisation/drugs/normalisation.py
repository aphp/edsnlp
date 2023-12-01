import pandas as pd
import numpy as np
import re
import spacy
import Levenshtein
from unidecode import unidecode
from tqdm import tqdm
import duckdb
from edsnlp.connectors import BratConnector
from collections import defaultdict
from exception import exception_list

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from levenpandas import fuzzymerge

class DrugNormaliser:
    def __init__(self, df_path, drug_dict, method="exact", max_pred=5, atc_len = 7):
        if df_path.endswith('json'):
            self.df = pd.read_json(df_path)
        else:
            self.df = self.gold_generation(df_path)
        self.drug_dict = drug_dict
        # self.df['drug'] = self.df['drug'].apply(lambda x: re.sub(r'\W+', '',x.lower()))
        self.df['norm_term'] = self.df['norm_term'].apply(lambda x: unidecode(x))
        # self.df['score'] = None

        merged_dict = {}
        for atc_code, values in self.drug_dict.items():
            # Shorten the ATC code
            shortened_code = atc_code[:atc_len]

            # Check if the shortened ATC code already exists in the merged dictionary
            if shortened_code in merged_dict:
                # Merge the arrays
                merged_dict[shortened_code] = list(set(merged_dict[shortened_code] + values))                    

            else:
                # Add a new entry for the shortened ATC code
                merged_dict[shortened_code] = values

        merged_dict = pd.DataFrame.from_dict({"norm_term": merged_dict}, "index").T.explode("norm_term").reset_index().rename(columns={"index": "label"})
        merged_dict.norm_term = merged_dict.norm_term.str.split(",")
        merged_dict = merged_dict.explode("norm_term").reset_index(drop=True)
        self.drug_dict = merged_dict
        self.method = method
        self.max_pred = max_pred
       
        
        
    
    def get_gold(self):
        return self.df
    
    def get_dict(self):
        return self.drug_dict
    
    def gold_generation(self, df_path):
        doc_list = BratConnector(df_path).brat2docs(spacy.blank("fr"))
        drug_list = []
        for doc in doc_list:
            for ent in doc.ents:
                if ent.label_ == 'Chemical_and_drugs':
                    if not ent._.Tech:
                        drug_list.append([ent.text, doc._.note_id, [ent.start, ent.end], ent.text.lower().strip()])

        drug_list_df = pd.DataFrame(drug_list, columns=['term', 'source', 'span_converted', 'norm_term'])
        drug_list_df.span_converted = drug_list_df.span_converted.astype(str)
        return drug_list_df
        

    
#     def exact_match(self, drug_name, atc, names):
#         matching_atc = []
#         matching_names = []
#         for name in names:
#             if drug_name == name:
#                 matching_atc.append(atc)
#                 matching_names.append(name)
#         return matching_atc, matching_names

#     def levenshtein_match(self, drug_name, name):
#         return Levenshtein.ratio(drug_name, name)
    
#     def dice_match(self, word1, word2):
#         intersection = len(set(word1) & set(word2))
#         coefficient = (2 * intersection) / (len(word1) + len(word2))
#         return coefficient


    def normalize(self, threshold=0.85):
        # self.df['pred_atc'] = [None]*len(self.df)
        # self.df['pred_string'] = [None]*len(self.df)

        for index, row in self.df.iterrows():
            for k, v in exception_list.items():
                if row["norm_term"] in v:
                    self.df.at[index, "norm_term"] = k

        if self.method == "exact":
            self.df = self.df.merge(self.drug_dict, how="left", on="norm_term")
        if self.method == "lev":
            df_1 = self.df.copy()
            df_2 = self.drug_dict.copy()
            merged_df = duckdb.query(
                f"""select *, jaro_winkler_similarity(df_1.norm_term, df_2.norm_term) score from df_1, df_2 where score > {threshold}"""
            ).to_df()
            idx = (
                merged_df.groupby(["term", "source", "span_converted", "norm_term"])[
                    "score"
                ].transform(max)
                == merged_df["score"]
            )
            self.df = merged_df[idx]
        self.df = self.df.groupby(
            ["term", "source", "span_converted", "norm_term"], as_index=False
        ).agg({"label": list})
        return self.df
        
#         if self.method =='lev':
#             for index, row in self.df.iterrows():
#                 drug_name = row['drug']
#                 matching_atc = []
#                 matching_names = []
#                 matching_scores = []
#                 for atc, names in self.drug_dict.items():
#                     names = [name for name in names if name is not np.nan]
#                     Levenshtein_distance = []
#                     Levenshtein_distance_name = []
#                     for name in names:
#                         Levenshtein_distance.append(self.levenshtein_match(drug_name,name))
#                         Levenshtein_distance_name.append(name)
#                     if len(Levenshtein_distance) > 0:
#                         max_value_id = Levenshtein_distance.index(max(Levenshtein_distance))
#                         max_value = Levenshtein_distance[max_value_id]
#                         max_value_name = Levenshtein_distance_name[max_value_id]
#                         if max_value >= treshold:
#                             matching_atc.append(atc)
#                             matching_names.append(max_value_name)
#                             matching_scores.append(max_value)
#                 #sort matching_atc by score
#                     matching_atc = [x for _,x in sorted(zip(matching_scores,matching_atc), reverse=True)]
#                     matching_names = [x for _,x in sorted(zip(matching_scores,matching_names), reverse=True)]
#                     matching_scores = sorted(matching_scores, reverse=True)
#                 if 1 in matching_scores:
#                     self.df.at[index, 'pred_atc'] = [matching_atc[i] for i, score in enumerate(matching_scores) if score == 1]
#                     self.df.at[index, 'pred_string'] = [matching_names[i] for i, score in enumerate(matching_scores) if score == 1]
#                     self.df.at[index, 'score'] = [score for score in matching_scores if score == 1]
#                 else:
#                     self.df.at[index, 'pred_atc'] = matching_atc[:self.max_pred]
#                     self.df.at[index, 'pred_string'] = matching_names[:self.max_pred]
#                     self.df.at[index, 'score'] = matching_scores[:self.max_pred]
#             return self.df

    

#     def acc(self, verbose = False):
#         correct_predictions = self.df.apply(lambda row: row['ATC'][:len(row['ATC'])] in [x[:len(row['ATC'])] for x in row['pred_atc']], axis=1).sum()
#         total_predictions = len(self.df)
#         accuracy = correct_predictions / total_predictions
#         if verbose:
#             return f'{accuracy}, ({correct_predictions}/{total_predictions})'
#         else:
#             return accuracy

#     def get_good_predictions(self):
#         good_predictions = self.df.apply(lambda row: row['ATC'][:len(row['ATC'])] in [x[:len(row['ATC'])] for x in row['pred_atc']], axis=1)
#         return self.df[good_predictions]
    
#     def get_bad_predictions(self):
#         bad_predictions = self.df.apply(lambda row: row['ATC'][:len(row['ATC'])] not in [x[:len(row['ATC'])] for x in row['pred_atc']], axis=1)
#         return self.df[bad_predictions]
    
#     def get_no_predictions(self):
#         no_predictions = self.df.apply(lambda row: len(row['pred_atc'])==0, axis=1)
#         return self.df[no_predictions]
            

#     def metrics(self, verbose = True):
#         y_true = self.df['ATC']
#         y_pred = self.df['pred_atc']

#         unique_atc = set([atc for atc in y_true])

#         results = {atc: {'TP': 0, 'FP': 0, 'FN': 0} for atc in unique_atc}

#         for atc in unique_atc:
#             TP = 0
#             FP = 0
#             FN = 0
#             for c,atc_gold in enumerate(y_true):
#                 if atc_gold == atc:
#                     if atc_gold in y_pred[c]:
#                         TP += 1
#                     else:
#                         FN += 1
#             for c, atcs_pred in enumerate(y_pred):
#                 for atc_pred in atcs_pred:
#                     if atc_pred == atc:
#                         if atc_pred not in y_true[c]:
#                             FP += 1

#             results[atc]['TP'] = TP
#             results[atc]['FP'] = FP
#             results[atc]['FN'] = FN
        
#         #we get the micro_average
#         total_TP = sum([results[atc]['TP'] for atc in unique_atc])
#         total_FP = sum([results[atc]['FP'] for atc in unique_atc])
#         total_FN = sum([results[atc]['FN'] for atc in unique_atc])

#         precision_micro = total_TP/(total_TP+total_FP)
#         recall_micro = total_TP/(total_TP+total_FN)
#         f1_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro)

#         #we get the macro_average
#         total_precision = 0
#         total_recall = 0
#         total_f1 = 0

#         for atc in unique_atc:
#             if results[atc]['TP']+results[atc]['FP'] != 0:
#                 precision = results[atc]['TP']/(results[atc]['TP']+results[atc]['FP'])
#             else:
#                 precision = 0
#             if results[atc]['TP']+results[atc]['FN'] != 0:
#                 recall = results[atc]['TP']/(results[atc]['TP']+results[atc]['FN'])
#             else:
#                 recall = 0
#             if precision+recall != 0:
#                 f1 = 2*precision*recall/(precision+recall)
#             else:
#                 f1 = 0
            
#             total_precision += precision
#             total_recall += recall
#             total_f1 += f1

#         total_precision = total_precision/len(unique_atc)
#         total_recall = total_recall/len(unique_atc)
#         total_f1 = total_f1/len(unique_atc)

       
#         print(f' MICRO : The precision is {precision_micro}, the recall is {recall_micro} and the f1 score is {f1_micro}')
#         print(f' MACRO : The precision is {total_precision}, the recall is {total_recall} and the f1 score is {total_f1}')
