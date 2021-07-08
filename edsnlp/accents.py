from spacy.tokens import Token, Doc
from unidecode import unidecode

if not Token.has_extension('ascii_'):
    Token.set_extension('ascii_', getter=lambda t: unidecode(t.text))

if not Token.has_extension('lowerascii_'):
    Token.set_extension('lowerascii_', getter=lambda t: unidecode(t.lower_))

if not Doc.has_extension('ascii_'):
    Doc.set_extension('ascii_', getter=lambda d: unidecode(d.text))

if not Doc.has_extension('lowerascii_'):
    Doc.set_extension('lowerascii_', getter=lambda d: unidecode(d.text).lower())
