import json
from typing import Any, Dict, List, Union

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.pipes.misc.relations import patterns


class RelationsMatcher:
    '''
    The `eds.relations` component links source and target Named Entities +/- Attributes.
    This component is rule-based and utilizes character proximity
    to determine relationships.


    Examples
    --------
    In this simple example, we extract drugs and dates \
        from a text and link them together.
    ```python
    import edsnlp, edsnlp.pipes as eds

    text = """
        Prise pendant 3 semaines d'Amlodipine 5mg per os une fois par jour \
            mais l'HTA reste mal contrôlée.
        Metformine 500 mg deux fois par jour à partir du 27/05/2022.
        Consultation chez un cardiologue le 11/07 pour évaluation de l'HTA, \
            dans l'attente majoration de l'AMLODIPINE à 10 mg.
        """

    scheme = {
        "source": [{"label": "drug", "attr": None}],
        "target": [
            {"label": "dates", "attr": None},
            {"label": "durations", "attr": None},
        ],
        "type": "Temporal",
        "inv_type": "inv_Temporal",
    }

    nlp = edsnlp.blank("eds")

    # Extraction of entities
    nlp.add_pipe("eds.drugs")
    nlp.add_pipe("eds.dates")
    # Extraction of sentences
    nlp.add_pipe("eds.sentences")
    # Extraction of relations
    nlp.add_pipe(
        "eds.relations",
        config={
            "scheme": scheme,
            "use_sentences": True,
            "clean_rel": True,
            "proximity_method": "sym",
            "max_dist": 60,
        },
    )
    doc = nlp(text)

    for label in doc.spans:
        print("Label: ", label, "\t Entities :", doc.spans[label])
        for span in doc.spans[label]:
            print("\t Entity :", span, "\t Relations :", span._.rel)

        # Out: Label:  drug 	 \
        # Entities : [Amlodipine, Metformine, AMLODIPINE]
        # Entity : Amlodipine 	 Relations : \
        # [{'type': 'Temporal', target': pendant 3 semaines}]
        # Entity : Metformine 	 Relations : \
        [{"type": "Temporal", "target": 27 / 05 / 2022}]
        # Entity : AMLODIPINE 	 Relations : []

        # Label:  dates 	 Entities : [27/05/2022, 11/07]
        # Entity : 27/05/2022 	 Relations : \
        [{"type": "inv_Temporal", "target": Metformine}]
        # Entity : 11/07 	 Relations : []

        # Label:  durations 	 Entities : [pendant 3 semaines]
        # Entity : pendant 3 semaines 	\
        Relations: [{"type": "inv_Temporal", "target": Amlodipine}]

    # Label:  periods 	 Entities : []
    ```

    Extensions
    ----------
    The `eds.relations` pipeline adds and declares one extension
    on the `Span` objects called `rel`. By default rel is an empty list.

    The `rel` extension is a list of dictionaries
    containing the type of the relation and the target `Span`.
    It automatically adds the inverse relation to the target `Span`.

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : str
        Name of the component
    scheme : Union[Union[Dict, List[Dict]],str]
        The scheme to use to match the relations
    use_sentences: bool = True
        Whether or not to use the `eds.sentences` matcher to improve results
    proximity_method: str = "right"
        The method to use to calculate the proximity between the entities
        "sym" : symmetrical distance
        "start" : distance between the start char of the entities
        "end" : distance between the end char of the entities
        "middle" : distance between the middle of the entities
        "right" : distance between the end of the source and the start of the target
        "left" : distance between the end of the target and the start of the source
    max_dist: int = 45
        The maximum distance between the entities to consider them as related
    clean_rel: bool = True
        Whether or not to clean the relations before adding new ones

    Scheme
    ------
    It can be a dictionary (one relation), \
        a list of dictionaries (one or more relations)
    or a string indicating the path of a json file.

    Each dictionary should contain the keys \
        `source`, `target`, `type` and `inv_type`.

    `source` and `target` are lists of dictionaries \
        containing the keys `label` and `attr`.

    `label` is the label of the entity to match.

    `attr` is a dictionary containing the attributes \
        to match on or None if no attribute is needed.

    `type` is the type of the relation.

    `inv_type` is the inverse type of the relation.
    ```json
    [
        {
            "source": [
                {
                    "label": "Chemical_and_drugs",
                    "attr": {
                        "Tech": [
                            null
                        ]
                    }
                }
            ],
            "target": [
                {
                    "label": "Temporal",
                    "attr": {
                        "AttTemp": [
                            "Duration",
                            "Date",
                            "Frequency",
                            "Time"
                        ]
                    }
                },
                {
                    "label": "Chemical_and_drugs",
                    "attr": {
                        "Tech": [
                            "dosage",
                            "route",
                            "strength",
                            "form"
                        ]
                    }
                }
            ],
            "type": "Depend",
            "inv_type": "inv_Depend"
        }
    ]
    ```

    Authors and citation
    --------------------
    The `eds.relations` was developed by AP-HP's Data Science team.

    '''

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str = "relations",
        *,
        scheme: Union[Union[Dict, List[Dict]], str] = None,
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
            if "source" not in scheme:
                raise ValueError("scheme must contain a 'source' key")
            if "target" not in scheme:
                raise ValueError("scheme must contain an 'target' key")
            if "type" not in scheme:
                raise ValueError("scheme must contain a 'type' key")
            if "inv_type" not in scheme:
                raise ValueError("scheme must contain an 'inv_type' key")
            if not isinstance(scheme["source"], list):
                raise ValueError("scheme['source'] must be a list")
            if not isinstance(scheme["target"], list):
                raise ValueError("scheme['target'] must be a list")
            if not isinstance(scheme["type"], str):
                raise ValueError("scheme['type'] must be a string")
            if not isinstance(scheme["inv_type"], str):
                raise ValueError("scheme['inv_type'] must be a string")
            for sub in scheme["source"]:
                if not isinstance(sub, dict):
                    raise ValueError("scheme['source'] must contain dictionaries")
                if "label" not in sub:
                    raise ValueError("scheme['source'] must contain a 'label' key")
                if not isinstance(sub["label"], str):
                    raise ValueError("scheme['source']['label'] must be a string")
                if "attr" in sub:
                    if sub["attr"] is not None and not isinstance(sub["attr"], dict):
                        raise ValueError(
                            "scheme['source']['attr'] must be a dictionary or None"
                        )
            for obj in scheme["target"]:
                if not isinstance(obj, dict):
                    raise ValueError("scheme['target'] must contain dictionaries")
                if "label" not in obj:
                    raise ValueError("scheme['target'] must contain a 'label' key")
                if not isinstance(obj["label"], str):
                    raise ValueError("scheme['target']['label'] must be a string")
                if "attr" in obj:
                    if obj["attr"] is not None and not isinstance(obj["attr"], dict):
                        raise ValueError(
                            "scheme['target']['attr'] must be a dictionary or None"
                        )
        return True

    @classmethod
    def set_extensions(cls) -> None:
        """Set the extension rel for the Span target."""
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
        """find the relations in the doc based \
            on the proximity of the entities attributes

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
                        if self.use_sentences and not self.sentences(
                            doc, span_obj["span"], span_sub["span"]
                        ):
                            continue
                        doc.spans[span_obj["label"]][span_obj["num_span"]]._.rel.append(
                            {
                                "type": dict_r[r]["inv_type"],
                                "target": doc.spans[span_sub["label"]][
                                    span_sub["num_span"]
                                ],
                            }
                        )

                        doc.spans[span_sub["label"]][span_sub["num_span"]]._.rel.append(
                            {
                                "type": dict_r[r]["type"],
                                "target": doc.spans[span_obj["label"]][
                                    span_obj["num_span"]
                                ],
                            }
                        )
        return doc

    def sentences(self, doc: Doc, span_obj: Span, span_sub: Span) -> bool:
        """Check if span_obj and span_sub are in the same sentence.

        Args:
            doc (Doc): EDSNLP Doc target
            span_obj (Span): span representing the target
            span_sub (Span): span representing the source

        Returns:
            bool: True if span_obj and span_sub \
                are in the same sentence, False otherwise.
        """
        for sent in doc.sents:
            if (
                span_obj.start >= sent.start
                and span_obj.end <= sent.end
                and span_sub.start >= sent.start
                and span_sub.end <= sent.end
            ):
                return True
        return False

    def find_relations(self, doc: Doc) -> Dict:
        """
        Detect the potential sources and targets in the document

        Args:
            doc (Doc): EDSNLP Doc target

        Returns:
            Dict: dict containing the potential sources and targets
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
            # Treatment of targets
            for obj in relation["target"]:
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
                                dict_r[r]["spans_obj"].append(
                                    {
                                        "label": label_obj,
                                        "num_span": num_span_obj,
                                        "span": span_obj,
                                    }
                                )
                    else:
                        for num_span_obj, span_obj in enumerate(doc.spans[label_obj]):
                            dict_r[r]["mat_obj"].append(
                                [span_obj.start_char, span_obj.end_char]
                            )
                            dict_r[r]["spans_obj"].append(
                                {
                                    "label": label_obj,
                                    "num_span": num_span_obj,
                                    "span": span_obj,
                                }
                            )

            # Treatment of sources
            for sub in relation["source"]:
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
                                dict_r[r]["spans_sub"].append(
                                    {
                                        "label": label_sub,
                                        "num_span": num_span_sub,
                                        "span": span_sub,
                                    }
                                )
                    else:
                        for num_span_sub, span_sub in enumerate(doc.spans[label_sub]):
                            dict_r[r]["mat_sub"].append(
                                [span_sub.start_char, span_sub.end_char]
                            )
                            dict_r[r]["spans_sub"].append(
                                {
                                    "label": label_sub,
                                    "num_span": num_span_sub,
                                    "span": span_sub,
                                }
                            )

            # Convert lists to numpy arrays for easier manipulation later
        for r in dict_r:
            dict_r[r]["mat_obj"] = np.array(dict_r[r]["mat_obj"])
            dict_r[r]["mat_sub"] = np.array(dict_r[r]["mat_sub"])

        return dict_r

    def calculate_min_distances(
        self, sources: NDArray[Any], targets: NDArray[Any]
    ) -> NDArray[Any]:
        """calculate the minimum distance between sources and targets

        Args:
            sources (NDArray[Any]): entities to be used as sources
            targets (NDArray[Any]): entities to be used as targets

        Returns:
            NDArray[Any]: the index of the source that is closest to the target
        """

        sources_expanded = sources[:, np.newaxis, :]
        targets_expanded = targets[np.newaxis, :, :]

        # calculate the distances between the entities
        distance_start_to_end = sources_expanded[:, :, 0] - targets_expanded[:, :, 1]
        distance_end_to_start = targets_expanded[:, :, 0] - sources_expanded[:, :, 1]
        distance_start_to_start = sources_expanded[:, :, 0] - targets_expanded[:, :, 0]
        distance_end_to_end = targets_expanded[:, :, 1] - sources_expanded[:, :, 1]
        distance_middle = (
            np.abs(distance_start_to_start) + np.abs(distance_end_to_end)
        ) / 2

        if self.proximity_method == "sym":
            distance_middle = np.abs(
                np.minimum(distance_start_to_start, distance_end_to_end)
            )
            # determine the mask for the left and right side of the entities
            mask_left = targets_expanded[:, :, 1] <= sources_expanded[:, :, 0]
            mask_right = sources_expanded[:, :, 1] <= targets_expanded[:, :, 0]

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
