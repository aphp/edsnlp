from typing import Optional, Any, Dict, List, Iterable
import time
from spacy.language import _copy_examples
from spacy.tokens import Doc
from spacy.training import validate_examples, Example
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from timeit import default_timer as timer

def get_annotation(docs):
    full_annot = []
    for doc in docs:
        annotation = [doc._.note_id]
        for label, ents in doc.spans.items():
            for ent in ents:
                annotation.append([ent.text, label, ent.start_char, ent.end_char])
        full_annot.append(annotation)
    return full_annot


def overlap(start_g, end_g, start_p, end_p, exact):
    if exact == False:
        if start_p <= start_g and end_p >= end_g:
            return 1
        else:
            return 0

    if exact == True:
        if start_g == start_p and end_g == end_p:
            return 1
        else:
            return 0
def compute_scores(
    ents_gold, ents_pred, boostrap_level="entity", exact=True, n_draw=500, alpha=0.05, digits=2
):
    docs = [doc[0] for doc in ents_gold]
    gold_labels = [
        [ent[1] for ent in doc[1:]] for doc in ents_gold
    ]  # get all the entities from the various documents of ents_gold
    gold_labels = set(
        [item for sublist in gold_labels for item in sublist]
    )  # flatten and transform it to a set to get unique values
    pred_labels = [
        [ent[1] for ent in doc[1:]] for doc in ents_pred
    ]  # get all the entities from the various documents of ents_gold
    pred_labels = set(
        [item for sublist in pred_labels for item in sublist]
    )  # flatten and transform it to a set to get unique values
    results = {  # we create a dic with the labels of the dataset (CHEM, BIO...)
        label: {} for label in pred_labels.union(gold_labels)
    }
    # COMPUTATION OF TRUE POSITIVE / FALSE POSITIVE / FALSE NEGATIVE
    for label in results.keys():
        results[label]["TP"] = 0
        results[label]["FP"] = 0
        results[label]["FN"] = 0
    results_by_doc = {doc : deepcopy(results) for doc in docs}
       
    for i in range(len(ents_gold)):  # iterate through doc
        # list of doc, inside each of them is a quadrupet ['text','label','start_char','stop_char']
        doc_id = ents_gold[i][0]
        ents_gold_doc = [(ent[1], ent[2], ent[3]) for ent in ents_gold[i][1:]]
        ents_pred_doc = [(ent[1], ent[2], ent[3]) for ent in ents_pred[i][1:]]

        for ent in ents_gold_doc:
            label_g = ent[0]
            start_g = ent[1]
            stop_g = ent[2]
            r = False

            for ent in ents_pred_doc:
                label_p = ent[0]
                start_p = ent[1]
                stop_p = ent[2]

                # exact is given as parameter because the overlap function take into account if we want an exact match or an inclusive match
                if (
                    label_g == label_p
                    and overlap(start_g, stop_g, start_p, stop_p, exact) > 0
                ):
                    r = True

            if r:
                results_by_doc[doc_id][label_g]["TP"] += 1
                results[label_g]["TP"] += 1
            else:
                results_by_doc[doc_id][label_g]["FN"] += 1
                results[label_g]["FN"] += 1

        for ent in ents_pred_doc:
            label_p = ent[0]
            start_p = ent[1]
            stop_p = ent[2]
            r = True

            for ent in ents_gold_doc:
                label_g = ent[0]
                start_g = ent[1]
                stop_g = ent[2]

                if (
                    label_g == label_p
                    and overlap(start_g, stop_g, start_p, stop_p, exact) > 0
                ):
                    r = False
            if r == True:
                results_by_doc[doc_id][label_p]["FP"] += 1
                results[label_p]["FP"] += 1

    if exact == True:
        print("Exact match")
    else:
        print("Inclusive match")

    # we will use this copy of the results dataframe
    results_list = deepcopy(results)

    # We transform the result dictionnary value from int to list to be able to append the new ones
    for key, value in results_list.items():
        for k, v in value.items():
            results_list[key][k] = [v]
                
    total_words = 0
    for entity in results_list.keys():
        total_words += results[entity]["TP"]
        total_words += results[entity]["FN"]
        total_words += results[entity]["FP"]
    label_to_draw = []
    proba = []
    for entity in results_list.keys():
        for test in ["TP", "FN", "FP"]:
            label_to_draw.append(entity + "-" + test)
            proba.append(results[entity][test] / total_words)

    micro_avg = {
        "TP": [sum(results[entity]["TP"] for entity in results.keys())],
        "FN": [sum(results[entity]["FN"] for entity in results.keys())],
        "FP": [sum(results[entity]["FP"] for entity in results.keys())],
    }
    
    # Bootstrap per doc
    if boostrap_level == 'doc':
        for i in tqdm(range(1, n_draw)):
            draw = np.random.choice(
                    docs,
                    size=len(docs),
                    replace=True,
                )
            micro_avg_draw = {"TP":0, "FN": 0, "FP": 0}
            results_draw = {  # we create a dic with the labels of the dataset (CHEM, BIO...)
                label: {} for label in pred_labels.union(gold_labels)
            }
            for label in results_draw.keys():
                results_draw[label]["Precision"] = 0
                results_draw[label]["TP"] = 0
                results_draw[label]["FP"] = 0
                results_draw[label]["FN"] = 0
            for doc in draw:
                for label in results_by_doc[doc].keys():
                    micro_avg_draw["TP"]+=results_by_doc[doc][label]["TP"]
                    results_draw[label]["TP"]+=results_by_doc[doc][label]["TP"]
                    micro_avg_draw["FN"]+=results_by_doc[doc][label]["FN"]
                    results_draw[label]["FN"]+=results_by_doc[doc][label]["FN"]
                    micro_avg_draw["FP"]+=results_by_doc[doc][label]["FP"]
                    results_draw[label]["FP"]+=results_by_doc[doc][label]["FP"]
            for entity in results_list.keys():
                results_list[entity]["TP"].append(
                    results_draw[entity]["TP"]
                )
                results_list[entity]["FN"].append(
                    results_draw[entity]["FN"]
                )
                results_list[entity]["FP"].append(
                    results_draw[entity]["FP"]
                )
            micro_avg["TP"].append(micro_avg_draw["TP"])
            micro_avg["FN"].append(micro_avg_draw["FN"])
            micro_avg["FP"].append(micro_avg_draw["FP"])
            
    # Bootstrap per entities
    if boostrap_level == 'entity':
        for i in tqdm(range(1, n_draw)):
            draw = np.random.choice(
                label_to_draw,
                size=total_words,
                p=proba,
                replace=True,
            )
            draw = np.stack(
                np.char.split(draw, "-"),
                axis=0,
            )
            micro_avg["TP"].append(len(draw[(draw[:, 1] == "TP")]))
            micro_avg["FN"].append(len(draw[(draw[:, 1] == "FN")]))
            micro_avg["FP"].append(len(draw[(draw[:, 1] == "FP")]))
            for entity in results_list.keys():
                results_list[entity]["TP"].append(
                    len(draw[(draw[:, 0] == entity) & (draw[:, 1] == "TP")])
                )
                results_list[entity]["FN"].append(
                    len(draw[(draw[:, 0] == entity) & (draw[:, 1] == "FN")])
                )
                results_list[entity]["FP"].append(
                    len(draw[(draw[:, 0] == entity) & (draw[:, 1] == "FP")])
                )

    results_list["Overall"] = micro_avg
    for entity in results_list.keys():
        results_list[entity]["N_entity"] = []
        results_list[entity]["Precision"] = []
        results_list[entity]["Recall"] = []
        results_list[entity]["F1"] = []
        for i in range(n_draw):
            results_list[entity]["N_entity"].append(results_list[entity]["TP"][i] + results_list[entity]["FP"][i] + results_list[entity]["FN"][i])
            if results_list[entity]["TP"][i] + results_list[entity]["FP"][i] != 0:
                results_list[entity]["Precision"].append(
                    results_list[entity]["TP"][i]
                    / (
                        results_list[entity]["TP"][i]
                        + results_list[entity]["FP"][i]
                    ) * 100
                )
            else:
                results_list[entity]["Precision"].append(int(results_list[entity]["TP"][i] == 0) * 100)
            if (results_list[entity]["TP"][i] + results_list[entity]["FN"][i]) != 0:
                results_list[entity]["Recall"].append(
                    results_list[entity]["TP"][i]
                    / (
                        results_list[entity]["TP"][i]
                        + results_list[entity]["FN"][i]
                    ) * 100
                )
            else:
                results_list[entity]["Recall"].append(int(results_list[entity]["TP"][i] == 0) * 100)
            if (
                results_list[entity]["Precision"][i]
                + results_list[entity]["Recall"][i]
            ) != 0:
                results_list[entity]["F1"].append(
                    2
                    * (
                        results_list[entity]["Precision"][i]
                        * results_list[entity]["Recall"][i]
                    )
                    / (
                        results_list[entity]["Precision"][i]
                        + results_list[entity]["Recall"][i]
                    )
                )
            else:
                results_list[entity]["F1"].append(0)
    # we aim at displaying the "true" observe value with confidence interval corresponding to the top 5 and 95% of the bootstrapped data
    lower_confidence_interval = {
        label: {
            k: round(np.quantile(v, alpha / 2), digits)
            for k, v in results_list[label].items()
            if k in ["Precision", "Recall", "F1", "N_entity"]
        }
        for label in results_list.keys()
    }
    upper_confidence_interval = {
        label: {
            k: round(np.quantile(v, (1 - alpha / 2)), digits)
            for k, v in results_list[label].items()
            if k in ["Precision", "Recall", "F1", "N_entity"]
        }
        for label in results_list.keys()
    }

    # we create a dict result_panel with the same keys as results_list but with the values of the nested dict being empty
    result_panel = {
        label: {
            k: ""
            for k, v in results_list[label].items()
            if k in ["Precision", "Recall", "F1", "N_entity"]
        }
        for label in results_list.keys()
    }

    # we take the value to build the result panel and the confidence interval
    # we take value['Precision'][0] because it is the original draw
    for key, value in results_list.items():
        precision = value["Precision"][0]
        precision_up = upper_confidence_interval[key]["Precision"]
        precision_down = lower_confidence_interval[key]["Precision"]
        recall = value["Recall"][0]
        recall_up = upper_confidence_interval[key]["Recall"]
        recall_down = lower_confidence_interval[key]["Recall"]
        f1 = value["F1"][0]
        f1_up = upper_confidence_interval[key]["F1"]
        f1_down = lower_confidence_interval[key]["F1"]
        n_entity = value["N_entity"][0]
        n_entity_up = upper_confidence_interval[key]["N_entity"]
        n_entity_down = lower_confidence_interval[key]["N_entity"]

        result_panel[key]["Precision"] = (
            str(round(precision, digits))
            + " ("
            + str(precision_down)
            + "-"
            + str(precision_up)
            + ")"
        )
        result_panel[key]["Recall"] = (
            str(round(recall, digits))
            + " ("
            + str(recall_down)
            + "-"
            + str(recall_up)
            + ")"
        )
        result_panel[key]["F1"] = (
            str(round(f1, digits)) + " (" + str(f1_down) + "-" + str(f1_up) + ")"
        )
        result_panel[key]["N_entity"] = (
            str(n_entity) + " (" + str(int(n_entity_up)) + "-" + str(int(n_entity_down)) + ")"
        )
    print(f"With alpha = {alpha} and {n_draw} draws")
    output = f"With alpha = {alpha} and {n_draw} draws\n"
    for key, value in result_panel.items():
        if "SECTION" not in key:
            output += f"\nLabel: {key}\n"
            for metric, metric_value in value.items():
                output += f"{metric}: {metric_value}\n"
            output += "-" * 30

    # print(output)
    result_panel["ents_per_type"] = {label: {"p": value["Precision"], "r": value["Recall"], "f": value["F1"], "n_entity": value["N_entity"]} for label, value in result_panel.items()}
    return result_panel

def evaluate_test(
    gold_docs: List[Doc],
    pred_docs: List[Doc],
    boostrap_level: str = "entity",
    exact: bool = True,
    n_draw: int = 500,
    alpha: float = 0.05,
    digits: int = 2,
) -> Dict[str, Any]:
    """
    Evaluate a model's pipeline components.

    Parameters
    ----------
    gold_docs : List[Doc]
        `Doc` objects.
    pred_docs : List[Doc]
        `Doc` objects.

    Returns
    -------
    Dict[str, Any]
        The evaluation results.
    """
    ents_pred, ents_gold = get_annotation(pred_docs), get_annotation(gold_docs)
    ents_pred.sort(key=lambda l: l[0])
    ents_gold.sort(key=lambda l: l[0])
    
    scores = compute_scores(
        ents_gold,
        ents_pred,
        boostrap_level=boostrap_level,
        exact=exact,
        n_draw=n_draw,
        alpha=alpha,
        digits=digits,
    )

    return scores


def evaluate(
    self,
    examples: Iterable[Example],
    *,
    batch_size: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Evaluate a model's pipeline components.

    Parameters
    ----------
    examples : Iterable[Example]
        `Example` objects.
    batch_size : Optional[int]
        Batch size to use.

    Returns
    -------
    Dict[str, Any]
        The evaluation results.
    """
    examples = list(examples)
    validate_examples(examples, "Language.evaluate")
    examples = _copy_examples(examples)
    if batch_size is None:
        batch_size = self.batch_size

    scores = {}

    total_time = 0

    begin_time = timer()
    # this is purely for timing
    for eg in examples:
        self.make_doc(eg.reference.text)
    total_time += timer() - begin_time

    n_words = sum(len(eg.predicted) for eg in examples)

    predictions = [eg.predicted for eg in examples]

    for name, component in self.pipeline:
        begin_time = timer()
        docs = [doc.copy() for doc in predictions]
        docs = list(component.pipe(docs, batch_size=batch_size))
        total_time += timer() - begin_time

        if name == "tok2vec":
            predictions = docs
        if hasattr(component, "score"):
            scores.update(
                component.score(
                    [Example(doc, eg.reference) for doc, eg in zip(docs, examples)]
                )
            )

    scores["speed"] = n_words / total_time
    return scores
