import context
import spacy
from pytest import fixture

import nlptools.components


@fixture(scope='session')
def nlp():
    model = spacy.blank('fr')
    
    model.add_pipe('sections')
    model.add_pipe('pollution')

    return model
