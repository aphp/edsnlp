from typing import List, Dict, Optional, Any, Union
from collections import defaultdict

import pandas as pd

import re

from loguru import logger

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spaczz.matcher import FuzzyMatcher

from edsnlp.rules.regex import RegexMatcher

from spacy.util import filter_spans

from edsnlp.rules.base import BaseComponent
from edsnlp.rules.generic import GenericMatcher

if not Span.has_extension('matcher_name'):
    Span.set_extension('matcher_name', default=None)

if not Span.has_extension('before_extract'):
    Span.set_extension('before_extract', default=None)
if not Span.has_extension('after_extract'):
    Span.set_extension('after_extract', default=None)

if not Span.has_extension('window'):
    Span.set_extension('window', default=None)

# todo: not used
if not Span.has_extension('before_snippet'):
    Span.set_extension('before_snippet', default=None)
if not Span.has_extension('after_snippet'):
    Span.set_extension('after_snippet', default=None)
    
class AdvancedRegex(GenericMatcher):
    def __init__(
            self,
            nlp: Language,
            regex_config: Dict[str, Any],
            window: int=10,
            verbose: int=0,
    ):
        
        self.regex_config = _check_regex_config(regex_config)
        self.window = window
        regex = {k:v['regex'] for k,v in regex_config.items()}
        attr = {k:v['attr'] for k,v in regex_config.items() if 'attr' in v}
        
        self.verbose = verbose
        
        super().__init__(nlp=nlp,
                         terms=dict(),
                         regex=regex,
                         attr=attr)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find matching spans in doc and apply some postprocessing
        
        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        sections:
            List of Spans referring to sections.
        """
        
        ents = super(AdvancedRegex, AdvancedRegex).process(self, doc)
        
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
        
        doc.ents += ents
        
        return doc
    
    def _postprocessing_pipeline(self, ents: List[Span]):
        # add a window within the sentence around entities
        ents = [self._add_window(ent) for ent in ents]
        
        # Remove entities based on the snippet located just before and after the entity
        ents = filter(self._exclude_filter, ents)
        
        # Extract informations from the entity's context via regex
        ents = [self._snippet_extraction(ent) for ent in ents]
        
        return tuple(ents)
    
    def _add_window(self, ent: Span) -> Span:
        ent._.window = ent.doc[max(ent.start-self.window, ent.sent.start) : min(ent.end+self.window, ent.sent.end)]
        ent._.before_snippet = ent.doc[max(ent.start-self.window, ent.sent.start) : ent.start]
        ent._.after_snippet = ent.doc[ent.end : min(ent.end+self.window, ent.sent.end)]
        return ent
    
    def _exclude_filter(self, ent: Span) -> Span:
        label = ent.label_
        
        before_exclude = self.regex_config[label].get('before_exclude', None)
        after_exclude = self.regex_config[label].get('after_exclude', None)
        
        if before_exclude is not None:
            before_snippet = ent.doc[ent._.window.start:ent.start+1]
            before_snippet = before_snippet._.norm if self.regex_config[label].get("attr")=="NORM" else before_snippet.text
            if re.compile(before_exclude).search(before_snippet) is not None:
                if self.verbose:
                    logger.info("excluded (before) string: " + str(before_snippet) + " - pattern: " + before_exclude)
                return False
            
        if after_exclude is not None:
            after_snippet = ent.doc[ent.end-1:ent._.window.end]
            after_snippet = after_snippet._.norm if self.regex_config[label].get("attr")=="NORM" else after_snippet.text
            if re.compile(after_exclude).search(after_snippet) is not None:
                if self.verbose:
                    logger.info("excluded (after) string: " + str(after_snippet) + " - pattern: " + after_exclude)
                return False
        
        return True
    
    def _snippet_extraction(self, ent: Span) -> Span:
        label = ent.label_
        
        before_extract = self.regex_config[label].get('before_extract', [])
        after_extract = self.regex_config[label].get('after_extract', [])
        
        if type(before_extract) == str:
            before_extract = [before_extract]
        if type(after_extract) == str:
            after_extract = [after_extract]
        
        # add 1 to ent.start so that we can extract the number when it is attached to the word, e.g. "3PA"
        before_snippet = ent.doc[ent._.window.start:ent.start+1] # todo: change tokenizer and remove +1 ?
        before_snippet = before_snippet._.norm if self.regex_config[label].get("attr")=="NORM" else before_snippet.text
        ent._.before_extract = []
        for pattern in before_extract:
            pattern = re.compile(pattern)
            match = pattern.search(before_snippet)
            ent._.before_extract.append(match.groups()[0] if match else None)
        
        after_snippet = ent.doc[ent.end-1:ent._.window.end]
        after_snippet = after_snippet._.norm if self.regex_config[label].get("attr")=="NORM" else after_snippet.text
        ent._.after_extract = []
        for pattern in after_extract:
            pattern = re.compile(pattern)
            match = pattern.search(after_snippet)
            ent._.after_extract.append(match.groups()[0] if match else None)
            

        return ent
        
def _check_regex_config(regex_config):
    for k,v in regex_config.items():
        if type(v) is not dict:
            raise TypeError(f"The value of the key {k} is of type {type(v)}, but a dict is expected")
        
        single_group_regex_keys = ['before_extract','after_extract']
        
        for single_group_regex_key in single_group_regex_keys:
            if single_group_regex_key in v:
                # ensure it is a list
                if type(v[single_group_regex_key]) is not list:
                    v[single_group_regex_key] = [v[single_group_regex_key]]
                
                for i, regex in enumerate(v[single_group_regex_key]):
                    n_groups = re.compile(regex).groups

                    if n_groups == 0:
                        # Adding grouping parenthesis
                        v[single_group_regex_key][i] = r'(' + regex + r')'
                    elif n_groups != 1:
                        # Accepting only 1 group per regex
                        raise ValueError(f"The RegEx for {repr(k)} ({repr(regex)}) stored in {repr(single_group_regex_key)} contains {n_groups} capturing groups, 1 expected")
                
    return regex_config