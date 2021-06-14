from spacy.tokens import Doc

import nlptools.accents

if not Doc.has_extension('note_id'):
    Doc.set_extension('note_id', default=None)
