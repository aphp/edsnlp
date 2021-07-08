from typing import List, Dict, Optional, Any, Union
from collections import defaultdict

import pandas as pd

import re

from loguru import logger

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spaczz.matcher import FuzzyMatcher

from nlptools.rules.regex import RegexMatcher

from spacy.util import filter_spans

from nlptools.rules.base import BaseComponent
from nlptools.rules.generic import GenericMatcher

if not Span.has_extension('matcher_name'):
    Span.set_extension('matcher_name', default=None)

if not Span.has_extension('before_extract'):
    Span.set_extension('before_extract', default=None)
if not Span.has_extension('after_extract'):
    Span.set_extension('after_extract', default=None)
if not Span.has_extension('before_snippet'):
    Span.set_extension('before_snippet', default=None)
if not Span.has_extension('after_snippet'):
    Span.set_extension('after_snippet', default=None)
    
class AdvancedRegex(GenericMatcher):
    """
    Provides a generic matcher component.

    Parameters
    ----------
    nlp:
        The Spacy object.
    terms:
        A dictionary of terms to look for.
    regex:
        A dictionary of regex patterns.
    fuzzy:
        Whether to perform fuzzy matching on the terms.
    fuzzy_kwargs:
        Default options for the fuzzy matcher, if used.
    filter_matches:
        Whether to filter out matches.
    """

    def __init__(
            self,
            nlp: Language,
            regex_config: Dict[str, Any],
            window: int=10
    ):
        
        self.regex_config = _check_regex_config(regex_config)
        self.window = window
        regex = {k:v['regex'] for k,v in regex_config.items()}
        
        super().__init__(nlp=nlp,
                         terms=dict(),
                         regex=regex)

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
        
        # Removing entities based on the snippet located just before and after the entity
        ents = [self._exclude_filter(ent) for ent in ents]
        
        # Extract informations from the entity's context via regex
        ents = [self._snippet_extraction(ent) for ent in ents if ent is not None]
        
        return tuple(ents)
        
    def _exclude_filter(self, ent: Span) -> Span:
        
        label = ent.label_
        
        before_exclude = self.regex_config[label].get('before_exclude', None)
        after_exclude = self.regex_config[label].get('after_exclude', None)
        
        to_keep = True
        
        if before_exclude is not None:
            before_exclude = re.compile(self.regex_config[label]['before_exclude'])
            before_snippet = ent.doc[max(0, ent.start - self.window):ent.start].text
            to_keep = to_keep & (before_exclude.search(before_snippet) is None)
            
        if after_exclude is not None:
            after_exclude = re.compile(self.regex_config[label]['after_exclude'])
            after_snippet = ent.doc[ent.end:ent.end + self.window].text
            to_keep = to_keep & (after_exclude.search(after_snippet) is None)
        
        return ent
    
    def _snippet_extraction(self, ent: Span) -> Span:
        
        label = ent.label_
        
        before_extract = self.regex_config[label].get('before_extract', None)
        after_extract = self.regex_config[label].get('after_extract', None)

        if before_extract is not None:
            before_extract = re.compile(self.regex_config[label]['before_extract'])
            before_snippet = ent.doc[max(0, ent.start - self.window):ent.start].text
            match = before_extract.search(before_snippet)
            ent._.before_extract = match.groups()[0] if match is not None else None
            
        if after_extract is not None:
            after_extract = re.compile(self.regex_config[label]['after_extract'])
            after_snippet = ent.doc[ent.end:ent.end + self.window].text
            match = after_extract.search(after_snippet)
            ent._.after_extract = match.groups()[0] if match is not None else None

        return ent
        
def _check_regex_config(regex_config):
    
    for k,v in regex_config.items():
        
        if type(v) is not dict:
            raise TypeError(f"The value of the key {k} is of type {type(v)}, but a dict is expected")
        
        single_group_regex_keys = ['before_extract','after_extract']
        
        for single_group_regex_key in single_group_regex_keys:
            regex = v.get(single_group_regex_key,'()')
            n_groups = re.compile(regex).groups
            
            if n_groups == 0:
                # Adding grouping parenthesis
                regex_config[k][single_group_regex_key] = r'(' + regex_config[k][single_group_regex_key] + r')'
            elif n_groups != 1:
                # Accepting only 1 group per regex
                raise ValueError(f"The RegEx for {repr(k)} ({repr(regex)}) stored in {repr(single_group_regex_key)} contains {n_groups} capturing groups, 1 expected")
                
    return regex_config