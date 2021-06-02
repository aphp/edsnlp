import context
import spacy
from pytest import fixture


@fixture(scope='session')
def nlp():
    model = spacy.blank('fr')
    
    model.add_pipe("sentencizer")
    model.add_pipe('section')
    model.add_pipe('pollution')

    return model
