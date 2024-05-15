from typing import Dict, Iterable, List, Union

from loguru import logger

from spacy.tokens import Doc, Span
from typing import Any
from numpy.typing import NDArray

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.misc.relations import patterns
import math as m
import numpy as np
import json


class RelationsMatcher:
    """ A spaCy EDSNLP pipeline component to find relations between entities based on their proximity.
    scheme = [
                    {
                        "subject": [{"label": "Chemical_and_drugs", "attr": {"Tech": [None]}}],
                        "object": [
                            {
                                "label": "Temporal",
                                "attr": {"AttTemp": [None, "Duration", "Date", "Time"]},
                            },
                            {
                                "label": "Chemical_and_drugs",
                                "attr": {"Tech": ["dosage", "route", "strength", "form"]},
                            },
                        ],
                        "type": "Depend",
                        "inv_type": "inv_Depend",
                    },
                ]
    """
    
    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str = "relations",
        *,
        scheme: Union[Union[Dict, List[Dict]],str] = None,
        use_sentences: bool = False,
        proximity_method: str = "right",
        clean_rel: bool = True,
        max_dist: int = 45,
    ):
        self.nlp = nlp
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        self.name = name

        if scheme is None:
            scheme = patterns.scheme
        if isinstance(scheme, str):
            #ouvrir le fichier json et le lire pour le mettre dans une variable
            if not scheme.endswith(".json"):
                raise ValueError("scheme must be a json file")
            with open(scheme) as f:
                scheme = json.load(f)
        if isinstance(scheme, dict):
            scheme = [scheme]
        self.check_scheme(scheme)
        self.scheme = scheme

        if not isinstance(use_sentences, bool):
            raise ValueError("use_sentences must be a boolean")
        self.use_sentences = use_sentences and (
            "eds.sentences" in nlp.pipe_names or "sentences" in nlp.pipe_names
        )
        if use_sentences and not self.use_sentences:
            logger.warning(
                "You have requested that the pipeline use annotations "
                "provided by the `eds.sentences` pipeline, but it was not set. "
                "Skipping that step."
            )

        if proximity_method not in ["sym", "start", "end", "middle", "right", "left"]:
            raise ValueError(
                """proximity_method must be one of 'sym','start',
                'end', 'middle', 'right', 'left'"""
            )
        self.proximity_method = proximity_method

        if not isinstance(clean_rel, bool):
            raise ValueError("clean_rel must be a boolean")
        self.clean_rel = clean_rel

        if not isinstance(max_dist, int):
            raise ValueError("max_dist must be an integer")
        self.max_dist = max_dist

        self.set_extensions()
    
    def check_scheme(self, schemes):
        for scheme in schemes:
            if not isinstance(scheme, dict):
                raise ValueError("scheme must be a dictionary")
            if "subject" not in scheme:
                raise ValueError("scheme must contain a 'subject' key")
            if "object" not in scheme:
                raise ValueError("scheme must contain an 'object' key")
            if "type" not in scheme:
                raise ValueError("scheme must contain a 'type' key")
            if "inv_type" not in scheme:
                raise ValueError("scheme must contain an 'inv_type' key")
            if not isinstance(scheme["subject"], list):
                raise ValueError("scheme['subject'] must be a list")
            if not isinstance(scheme["object"], list):
                raise ValueError("scheme['object'] must be a list")
            if not isinstance(scheme["type"], str):
                raise ValueError("scheme['type'] must be a string")
            if not isinstance(scheme["inv_type"], str):
                raise ValueError("scheme['inv_type'] must be a string")
            for sub in scheme["subject"]:
                if not isinstance(sub, dict):
                    raise ValueError("scheme['subject'] must contain dictionaries")
                if "label" not in sub:
                    raise ValueError("scheme['subject'] must contain a 'label' key")
                if not isinstance(sub["label"], str):
                    raise ValueError("scheme['subject']['label'] must be a string")
                if "attr" in sub:
                    if sub["attr"] is not None and not isinstance(sub["attr"], dict):
                        raise ValueError("scheme['subject']['attr'] must be a dictionary or None")
            for obj in scheme["object"]:
                if not isinstance(obj, dict):
                    raise ValueError("scheme['object'] must contain dictionaries")
                if "label" not in obj:
                    raise ValueError("scheme['object'] must contain a 'label' key")
                if not isinstance(obj["label"], str):
                    raise ValueError("scheme['object']['label'] must be a string")
                if "attr" in obj:
                    if obj["attr"] is not None and not isinstance(obj["attr"], dict):
                        raise ValueError("scheme['object']['attr'] must be a dictionary or None")
        return True

    @classmethod
    def set_extensions(cls) -> None:
        """Set the extension rel for the Span object.
        """
        if not Span.has_extension("rel"):
            Span.set_extension("rel", default=[])
    
    def clean_relations(self, doc: Doc) -> Doc:
        """Remove the relations from the doc

        Args:
            doc (Doc): the doc to be processed

        Returns:
            Doc: the doc with the relations removed
        """
        for label, spans in doc.spans.items():
            for span in spans:
                if span._.rel:
                    span._.rel = []
        return doc

    def __call__(self, doc: Doc) -> Doc:
        """find the relations in the doc based on the proximity of the entities attributes

        Args:
            doc (Doc): the doc to be processed

        Returns:
            Doc: the doc with the relations added
        """
        if self.clean_rel:
            doc = self.clean_relations(doc)

        dict_r = self.find_relations(doc)

        for r, rel in enumerate(dict_r):
            if len(dict_r[r]["mat_obj"]) > 0 and len(dict_r[r]["mat_sub"]) > 0:
                min_distance_indices, distances = self.calculate_min_distances(
                    dict_r[r]["mat_sub"], dict_r[r]["mat_obj"]
                )
                for i, span_obj in enumerate(dict_r[r]["spans_obj"]):
                    if distances[min_distance_indices[i]][i] <= self.max_dist:
                        span_sub = dict_r[r]["spans_sub"][min_distance_indices[i]]
                        if self.use_sentences and not self.sentences(doc, span_obj["span"], span_sub["span"]):
                            continue
                        doc.spans[span_obj['label']][span_obj['num_span']]._.rel.append(
                                    {"type": dict_r[r]["inv_type"], "target": doc.spans[span_sub["label"]][span_sub["num_span"]]}
                                )

                        doc.spans[span_sub['label']][span_sub['num_span']]._.rel.append(
                                    {"type": dict_r[r]["type"], "target": doc.spans[span_obj["label"]][span_obj["num_span"]]}
                                )
        return doc
    
    def sentences(self, doc: Doc, span_obj: Span, span_sub: Span) -> bool:
        """ Check if span_obj and span_sub are in the same sentence.

        Args:
            doc (Doc): EDSNLP Doc object
            span_obj (Span): span representing the target
            span_sub (Span): span representing the source

        Returns:
            bool: True if span_obj and span_sub are in the same sentence, False otherwise.
        """
        for sent in doc.sents:
            if span_obj.start >= sent.start and span_obj.end <= sent.end and span_sub.start >= sent.start and span_sub.end <= sent.end:
                return True
        return False


    def find_relations(self, doc: Doc) -> Dict:
        """
        Detect the potential subjects and objects in the document

        Args:
            doc (Doc): EDSNLP Doc object

        Returns:
            Dict: dict containing the potential subjects and objects
        """
        dict_r = {}
        for r, relation in enumerate(self.scheme):
            dict_r[r] = {
                "mat_obj": [],
                "spans_obj": [],
                "mat_sub": [],
                "spans_sub": [],
                "type": relation["type"],
                "inv_type": relation["inv_type"],
            }
            # Treatment of objects
            for obj in relation["object"]:
                label_obj = obj["label"]
                attr_obj = obj["attr"]
                if label_obj in doc.spans:
                    if attr_obj is not None:
                        for num_span_obj, span_obj in enumerate(doc.spans[label_obj]):
                            if self.filter_spans(
                                span_obj, *list(attr_obj.items())[0], label_obj
                            ):
                                dict_r[r]["mat_obj"].append(
                                    [span_obj.start_char, span_obj.end_char]
                                )
                                dict_r[r]["spans_obj"].append({'label': label_obj, 'num_span': num_span_obj, 'span': span_obj})
                    else:
                        for num_span_obj, span_obj in enumerate(doc.spans[label_obj]):
                            dict_r[r]["mat_obj"].append(
                                [span_obj.start_char, span_obj.end_char]
                            )
                            dict_r[r]["spans_obj"].append({'label': label_obj, 'num_span': num_span_obj, 'span': span_obj})

            # Treatment of subjects
            for sub in relation["subject"]:
                label_sub = sub["label"]
                attr_sub = sub["attr"]
                if label_sub in doc.spans:
                    if attr_sub is not None:
                        for num_span_sub, span_sub in enumerate(doc.spans[label_sub]):
                            if self.filter_spans(
                                span_sub, *list(attr_sub.items())[0], label_sub
                            ):
                                dict_r[r]["mat_sub"].append(
                                    [span_sub.start_char, span_sub.end_char]
                                )
                                dict_r[r]["spans_sub"].append({'label': label_sub, 'num_span': num_span_sub, 'span': span_sub})
                    else:
                        for num_span_sub, span_sub in enumerate(doc.spans[label_sub]):
                            dict_r[r]["mat_sub"].append(
                                [span_sub.start_char, span_sub.end_char]
                            )
                            dict_r[r]["spans_sub"].append({'label': label_sub, 'num_span': num_span_sub, 'span': span_sub})

            # Convert lists to numpy arrays for easier manipulation later
        for r in dict_r:
            dict_r[r]["mat_obj"] = np.array(dict_r[r]["mat_obj"])
            dict_r[r]["mat_sub"] = np.array(dict_r[r]["mat_sub"])

        return dict_r

    def calculate_min_distances(self, subjects:NDArray[Any], objects:NDArray[Any]) -> NDArray[Any]:
        """ calculate the minimum distance between subjects and objects

        Args:
            subjects (NDArray[Any]): entities to be used as subjects
            objects (NDArray[Any]): entities to be used as objects

        Returns:
            NDArray[Any]: the index of the subject that is closest to the object
        """

        subjects_expanded = subjects[:, np.newaxis, :]
        objects_expanded = objects[np.newaxis, :, :]

        # calculate the distances between the entities
        distance_start_to_end = subjects_expanded[:, :, 0] - objects_expanded[:, :, 1]
        distance_end_to_start = objects_expanded[:, :, 0] - subjects_expanded[:, :, 1]
        distance_start_to_start = subjects_expanded[:, :, 0] - objects_expanded[:, :, 0]
        distance_end_to_end = objects_expanded[:, :, 1] - subjects_expanded[:, :, 1]
        distance_middle = (
            subjects_expanded[:, :, 0] + subjects_expanded[:, :, 1]
        ) / 2 - (objects_expanded[:, :, 0] + objects_expanded[:, :, 1]) / 2

        if self.proximity_method == "sym":
            distance_middle = np.abs(
                np.minimum(distance_start_to_start, distance_end_to_end)
            )
            # determine the mask for the left and right side of the entities
            mask_left = objects_expanded[:, :, 1] <= subjects_expanded[:, :, 0]
            mask_right = subjects_expanded[:, :, 1] <= objects_expanded[:, :, 0]

            # Assign the distances based on the mask
            distances = np.where(mask_left, np.abs(distance_start_to_end), np.inf)
            distances = np.where(mask_right, np.abs(distance_end_to_start), distances)
            mask_middle = np.logical_not(np.logical_or(mask_left, mask_right))
            distances = np.where(mask_middle, distance_middle, distances)
            distances = np.where(distances == 0, np.inf, distances)

            # find the index of the minimum distance
            min_distance_indices = np.argmin(distances, axis=0)

        if self.proximity_method == "right":
            distances = np.abs(distance_end_to_start)
            distances = np.where(distances == 0, np.inf, distances)
            min_distance_indices = np.argmin(distances, axis=0)

        if self.proximity_method == "left":
            distances = np.abs(distance_start_to_end)
            distances = np.where(distances == 0, np.inf, distances)
            min_distance_indices = np.argmin(distances, axis=0)

        if self.proximity_method == "start":
            distances = np.abs(distance_start_to_start)
            distances = np.where(distances == 0, np.inf, distances)
            min_distance_indices = np.argmin(distances, axis=0)

        if self.proximity_method == "end":
            distances = np.abs(distance_end_to_end)
            distances = np.where(distances == 0, np.inf, distances)
            min_distance_indices = np.argmin(distances, axis=0)

        if self.proximity_method == "middle":
            distances = distance_middle
            distances = np.where(distances == 0, np.inf, distances)
            min_distance_indices = np.argmin(distances, axis=0)

        return min_distance_indices, distances
    
    def filter_spans(
        self, span: Span, attr_name: str, attr_values: list, label: str
    ) -> bool:
        """Filter the spans based on the attribute values

        Args:
            span (Span): the span to be filtered
            attr_name (str): the name of the attribute
            attr_values (list): the values of the attribute
            label (str): the label of the span

        Returns:
            bool: _description_
        """
        # Get the attribute value or None if it doesn't exist
        attr_value = getattr(span._, attr_name, None)  
        if attr_value in attr_values:
            return True
        return False


