from spacy.tokens import Token

if not Token.has_extension("excluded"):
    Token.set_extension("excluded", default=False)


def excluded_or_space_getter(t):
    return t.is_space or t.tag_ == "EXCLUDED"


if not Token.has_extension("excluded_or_space"):
    Token.set_extension(
        "excluded_or_space",
        getter=excluded_or_space_getter,
    )
