import re
from typing import Any, Dict, List, Optional

import spacy
from loguru import logger
from spacy import registry
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.core.matcher import GenericMatcher
from edsnlp.utils.filter import filter_spans

FILTER_PROCESS_KEY = "ent_filter_process"

ALLOWED_KEYS = [
    "regex",
    "before_extract",
    "after_extract",
    "before_exclude",
    "after_exclude",
    "window",
    "attr",
    "ent_filter_process",
]


class AdvancedRegex(GenericMatcher):
    """
    Allows additional matching in the surrounding context of the main match group,
    for qualification/filtering.

    Parameters
    ----------
    nlp : Language
        spaCy `Language` object.
    regex_config : Dict[str, Any]
        Configuration for the main expression.
        A dictionary of the form pattern_name -> pattern.
        A pattern is a dictionary of the form key -> value
        The available keys are:
        - regex:
            the pattern that defines an entity
        - before_extract, after_extract:
            pattern or list of patterns
            the results are stored as a list in e._.before_extract and after_extract
        - before_exclude, after_exclude:
            pattern or list of patterns.
            If a pattern matches within the window around the entity,
            the entity is discarded
        - window:
            overrides default
        - attr:
            overrides default
        - ent_filter_process:
            The key is expected to hold the name of a
            fonction registred in spacy's registry["misc"].
            The expected signature of the function is Span -> Optional[Span].
            It is called to update each entity found.
            If the function returns None, the entity is filtered out.
    window : int
        Number of tokens to consider before and after the main expression.
        Each pattern can override this default value.
    attr : str
        Attribute to match on, eg `TEXT`, `NORM`, etc.
        Each pattern can override this default value.
    verbose : int
        Verbosity level, useful for debugging.
    ignore_excluded : bool
        Whether to skip excluded tokens.
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        regex_config: Dict[str, Dict[str, Any]],
        window: Optional[int],
        attr: Optional[str],
        verbose: int,
        ignore_excluded: bool,
    ):
        regex_config = _check_regex_config(regex_config)
        self.regex_config = self._load_functions(regex_config)

        self.window = window
        self.name = name
        regex = regex_config

        self.verbose = verbose

        super().__init__(
            nlp=nlp,
            terms=dict(),
            regex=regex,
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        self.ignore_excluded = ignore_excluded

        self.set_extensions()

    @staticmethod
    def set_extensions() -> None:
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

    def _load_functions(self, regex_config):
        for lab, conf in regex_config.items():
            if FILTER_PROCESS_KEY in conf:
                conf[FILTER_PROCESS_KEY] = registry.get(
                    "misc", conf[FILTER_PROCESS_KEY]
                )
        return regex_config

    def process(self, doc: Doc) -> List[Span]:
        """
        Process the document, looking for named entities.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

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
        doc.spans[self.name] = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc

    def _postprocessing_pipeline(self, ents: List[Span]):
        # add a window within the sentence around entities
        ents = [self._add_window(ent) for ent in ents]

        # Remove entities based on the snippet located just before and after the entity
        ents = filter(self._exclude_filter, ents)

        # Extract information from the entity's context via regex
        ents = [self._snippet_extraction(ent) for ent in ents]

        # apply the function 'filter_process' if defined
        ents = self._apply_filter_process(ents)

        return ents

    def _add_window(self, ent: Span) -> Span:
        window = self.regex_config[ent.label_].get("window", self.window)

        ent._.window = ent.doc[
            max(ent.start - window, ent.sent.start) : min(
                ent.end + window, ent.sent.end
            )
        ]

        # include the entity in the snippets so that we can extract
        # the number when it is attached to the word, e.g. "3PA"
        ent._.before_snippet = ent.doc[
            max(ent.start - window, ent.sent.start) : ent.end
        ]
        ent._.after_snippet = ent.doc[ent.start : min(ent.end + window, ent.sent.end)]
        return ent

    def get_text(self, span: Span, label) -> str:
        attr = self.regex_config[label].get("attr", self.attr)

        return get_text(
            doclike=span,
            attr=attr,
            ignore_excluded=self.ignore_excluded,
        )

    def _exclude_filter(self, ent: Span) -> Span:
        label = ent.label_

        before_exclude = self.regex_config[label].get("before_exclude", None)
        after_exclude = self.regex_config[label].get("after_exclude", None)

        if before_exclude is not None:
            t = ent._.before_snippet
            t = self.get_text(t, label)
            if re.compile(before_exclude).search(t) is not None:
                if self.verbose:
                    logger.info(
                        f"excluded (before) string: {t} - pattern {before_exclude}"
                    )
                return False

        if after_exclude is not None:
            t = ent._.after_snippet
            t = self.get_text(t, label)
            if re.compile(after_exclude).search(t) is not None:
                if self.verbose:
                    logger.info(
                        f"excluded (after) string: {t} - pattern {after_exclude}"
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
        t = self.get_text(t, label)
        ent._.before_extract = []
        for pattern in before_extract:
            pattern = re.compile(pattern)
            match = pattern.search(t)
            ent._.before_extract.append(match.groups()[0] if match else None)

        t = ent._.after_snippet
        t = self.get_text(t, label)
        ent._.after_extract = []
        for pattern in after_extract:
            pattern = re.compile(pattern)
            match = pattern.search(t)
            ent._.after_extract.append(match.groups()[0] if match else None)

        return ent

    def _apply_filter_process(self, ents):
        new_ents = []
        for e in ents:
            if FILTER_PROCESS_KEY in self.regex_config[e.label_]:
                new_e = self.regex_config[e.label_][FILTER_PROCESS_KEY](e)
                if new_e is not None:
                    new_ents.append(new_e)
            else:
                new_ents.append(e)

        return new_ents


def _check_regex_config(regex_config):
    for pat, v in regex_config.items():
        for k in v:
            assert k in ALLOWED_KEYS, f"key {k} in pattern {pat} not recognized"

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
                            f"The RegEx for {repr(k)} ({repr(regex)}) "
                            f"stored in {repr(single_group_regex_key)} "
                            f"contains {n_groups} capturing groups, 1 expected"
                        )

    return regex_config


def register_patterns_functions(
    patterns: Dict[str, Dict[str, Any]],
    name: str,
    function_key="ent_filter_process",
):
    """
    For each pattern, store the function located at `function_key` in spaCy's registry.
    In the input dictionary, replace the function by its identifier.

    Parameters
    ----------
    patterns:
        The dictionary of named patterns.

    name:
        The name of the pipeline. Used to resolve naming issues.

    function_key:
        The key in each pattern where the function is stored.

    Returns
    -------
    patterns:
        the patterns where the functions have been replaced by their identifiers.
    """
    for pattern_name, pattern in patterns.items():
        if function_key in pattern:
            identifier = "__".join([name, function_key, pattern_name])
            spacy.registry.misc(identifier, pattern[function_key])
            pattern[function_key] = identifier
    return patterns
