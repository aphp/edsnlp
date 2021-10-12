import re
from typing import Any, Dict, List

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.matcher import GenericMatcher
from edsnlp.utils.filter import filter_spans

if not Doc.has_extension("my_ents"):
    Doc.set_extension("my_ents", default=[])

if not Span.has_extension("matcher_name"):
    Span.set_extension("matcher_name", default=None)

if not Span.has_extension("before_extract"):
    Span.set_extension("before_extract", default=None)
if not Span.has_extension("after_extract"):
    Span.set_extension("after_extract", default=None)

if not Span.has_extension("window"):
    Span.set_extension("window", default=None)

if not Span.has_extension("before_snippet"):
    Span.set_extension("before_snippet", default=None)
if not Span.has_extension("after_snippet"):
    Span.set_extension("after_snippet", default=None)


class AdvancedRegex(GenericMatcher):
    def __init__(
        self,
        nlp: Language,
        regex_config: Dict[str, Any],
        window: int,
        verbose: int,
    ):

        self.regex_config = _check_regex_config(regex_config)
        self.window = window
        regex = {k: v["regex"] for k, v in regex_config.items()}
        attr = {k: v["attr"] for k, v in regex_config.items() if "attr" in v}

        self.verbose = verbose

        super().__init__(
            nlp=nlp,
            terms=dict(),
            regex=regex,
            attr=attr,
            fuzzy=False,
            fuzzy_kwargs=None,
            filter_matches=True,
            on_ents_only=False,
        )

    def process(self, doc: Doc) -> List[Span]:
        """
        Process the document, looking for named entities.

        Parameters
        ----------
        doc : Doc
            Spacy Doc object

        Returns
        -------
        List[Span]
            List of detected spans.
        """

        ents = super().process(doc)
        ents = self._postprocessing_pipeline(ents)

        return ents

    def __call__(self, doc: Doc) -> Doc:
        """
        Adds spans to document.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for extracted terms.
        """

        ents = self.process(doc)

        ents, discarded = filter_spans(list(doc.ents) + ents, return_discarded=True)

        doc.ents = ents
        doc.spans["discarded"] = discarded

        return doc

    def _postprocessing_pipeline(self, ents: List[Span]):
        # add a window within the sentence around entities
        ents = [self._add_window(ent) for ent in ents]

        # Remove entities based on the snippet located just before and after the entity
        ents = filter(self._exclude_filter, ents)

        # Extract informations from the entity's context via regex
        ents = [self._snippet_extraction(ent) for ent in ents]

        return ents

    def _add_window(self, ent: Span) -> Span:
        ent._.window = ent.doc[
            max(ent.start - self.window, ent.sent.start) : min(
                ent.end + self.window, ent.sent.end
            )
        ]

        # include the entity in the snippets so that we can extract the number when it is attached to the word, e.g. "3PA"
        ent._.before_snippet = ent.doc[
            max(ent.start - self.window, ent.sent.start) : ent.end
        ]
        ent._.after_snippet = ent.doc[
            ent.start : min(ent.end + self.window, ent.sent.end)
        ]
        return ent

    def _exclude_filter(self, ent: Span) -> Span:
        label = ent.label_

        before_exclude = self.regex_config[label].get("before_exclude", None)
        after_exclude = self.regex_config[label].get("after_exclude", None)

        if before_exclude is not None:
            t = ent._.before_snippet
            t = (
                t._.norm
                if self.regex_config[label].get("attr", self.DEFAULT_ATTR) == "NORM"
                else t.text
            )
            if re.compile(before_exclude).search(t) is not None:
                if self.verbose:
                    logger.info(
                        "excluded (before) string: "
                        + str(t)
                        + " - pattern: "
                        + before_exclude
                    )
                return False

        if after_exclude is not None:
            t = ent._.after_snippet
            t = (
                t._.norm
                if self.regex_config[label].get("attr", self.DEFAULT_ATTR) == "NORM"
                else t.text
            )
            if re.compile(after_exclude).search(t) is not None:
                if self.verbose:
                    logger.info(
                        "excluded (after) string: "
                        + str(t)
                        + " - pattern: "
                        + after_exclude
                    )
                return False

        return True

    def _snippet_extraction(self, ent: Span) -> Span:
        label = ent.label_

        before_extract = self.regex_config[label].get("before_extract", [])
        after_extract = self.regex_config[label].get("after_extract", [])

        if type(before_extract) == str:
            before_extract = [before_extract]
        if type(after_extract) == str:
            after_extract = [after_extract]

        t = ent._.before_snippet
        t = (
            t._.norm
            if self.regex_config[label].get("attr", self.DEFAULT_ATTR) == "NORM"
            else t.text
        )
        ent._.before_extract = []
        for pattern in before_extract:
            pattern = re.compile(pattern)
            match = pattern.search(t)
            ent._.before_extract.append(match.groups()[0] if match else None)

        t = ent._.after_snippet
        t = (
            t._.norm
            if self.regex_config[label].get("attr", self.DEFAULT_ATTR) == "NORM"
            else t.text
        )
        ent._.after_extract = []
        for pattern in after_extract:
            pattern = re.compile(pattern)
            match = pattern.search(t)
            ent._.after_extract.append(match.groups()[0] if match else None)

        return ent


def _check_regex_config(regex_config):
    for k, v in regex_config.items():
        if type(v) is not dict:
            raise TypeError(
                f"The value of the key {k} is of type {type(v)}, but a dict is expected"
            )

        single_group_regex_keys = ["before_extract", "after_extract"]

        for single_group_regex_key in single_group_regex_keys:
            if single_group_regex_key in v:
                # ensure it is a list
                if type(v[single_group_regex_key]) is not list:
                    v[single_group_regex_key] = [v[single_group_regex_key]]

                for i, regex in enumerate(v[single_group_regex_key]):
                    n_groups = re.compile(regex).groups

                    if n_groups == 0:
                        # Adding grouping parenthesis
                        v[single_group_regex_key][i] = r"(" + regex + r")"
                    elif n_groups != 1:
                        # Accepting only 1 group per regex
                        raise ValueError(
                            f"The RegEx for {repr(k)} ({repr(regex)}) stored in {repr(single_group_regex_key)} "
                            f"contains {n_groups} capturing groups, 1 expected"
                        )

    return regex_config
