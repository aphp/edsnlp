from edsnlp.utils.regex import make_pattern

# Phones
delimiters = ["", r"\.", r"\-", " "]
phone_pattern = make_pattern(
    [r"((\+33|0033)|0[1234568]) ?" + d.join([r"\d{2}"] * 4) for d in delimiters]
)

# IPP
ipp_pattern = r"\b(8\d{9})\b"

# NDA
nda_pattern = (
    r"\b((?:01|05|09|10|11|14|16|21|22|26|28|29|32|33|36|38|41|42|44|47"
    r"|49|53|61|64|66|68|69|72|73|76|79|84|87|88|90|95|96|99|AG)\d{7,8})\b"
)

# Mail
mail_pattern = r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"


# Zip
zip_pattern = r"(\b(?:(?:[0-8][0-9])|(9[0-5])|(2[abAB]))[0-9]{3})[^\d]"

# NSS
nss_pattern = (
    r"((?:[1-37-8])(?:[0-9]{2})(?:0[0-9]|[2-35-9][0-9]|[14][0-2])"
    r"(?:(?:0[1-9]|[1-8][0-9]|9[0-69]|2[abAB])(?:00[1-9]|0[1-9][0-9]|"
    r"[1-8][0-9]{2}|9[0-8][0-9]|990)|(?:9[78][0-9])(?:0[1-9]|[1-8][0-9]|90))"
    r"(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2})(?:0[1-9]|[1-8][0-9]|9[0-7]))"
)


patterns = dict(
    # ADRESSE
    # DATE
    # DATE_NAISSANCE
    # HOPITAL
    IPP=ipp_pattern,
    MAIL=mail_pattern,
    NDA=nda_pattern,
    # NOM
    # PRENOM
    SECU=nss_pattern,
    TEL=phone_pattern,
    # VILLE
    ZIP=zip_pattern,
)
