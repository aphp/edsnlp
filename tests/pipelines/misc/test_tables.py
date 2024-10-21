import pytest
from spacy.tokens.span import Span

TEXT = """
Le patientqsfqfdf bla bla bla
Leucocytes ¦x10*9/L ¦4.97 ¦4.09-11
Hématies ¦x10*12/L¦4.68 ¦4.53-5.79
Hémoglobine ¦g/dL ¦14.8 ¦13.4-16.7
Hématocrite ¦% ¦44.2 ¦39.2-48.6
VGM ¦fL ¦94.4 + ¦79.6-94
TCMH ¦pg ¦31.6 ¦27.3-32.8
CCMH ¦g/dL ¦33.5 ¦32.4-36.3
Plaquettes ¦x10*9/L ¦191 ¦172-398
VMP ¦fL ¦11.5 + ¦7.4-10.8

qdfsdf

2/2Pat : <NOM> <Prenom> |<date> | <ipp> |Intitulé RCP

Table de taille <= 3 :

 |Libellé | Unité | Valeur | Intervalle |
 |Leucocytes |x10*9/L |4.97 | 4.09-11 |

qdfsdf

 |Libellé | Unité | Valeur | Intervalle |
 |Leucocytes |x10*9/L |4.97 | 4.09-11 |
 |Hématies |x10*12/L|4.68 | 4.53-5.79 |
 |Hémoglobine |g/dL |14.8 | 13.4-16.7 |
 |Hématocrite ||44.2 | 39.2-48.6 |
 |VGM |fL | 94.4 + | 79.6-94 |
 |TCMH |pg |31.6 |
 |CCMH |g/dL
 |Plaquettes |x10*9/L |191 | 172-398 |
 |VMP |fL |11.5 + | 7.4-10.8 |

"""


def test_tables(blank_nlp):
    if blank_nlp.lang != "eds":
        pytest.skip("Test only for eds language")
    blank_nlp.add_pipe("eds.normalizer")
    blank_nlp.add_pipe("eds.tables", config=dict(min_rows=3))

    doc = blank_nlp(TEXT)

    assert len(doc.spans["tables"]) == 2

    span = doc.spans["tables"][0]
    df = span._.to_pd_table()
    assert len(df.columns) == 4
    assert len(df) == 9
    assert str(df.iloc[5, 0]) == "TCMH"

    span = doc.spans["tables"][1]
    df = span._.to_pd_table(header=True, index=True, as_spans=True)
    assert df.columns.tolist() == [
        "Unité",
        "Valeur",
        "Intervalle",
    ]
    assert df.index.tolist() == [
        "Leucocytes",
        "Hématies",
        "Hémoglobine",
        "Hématocrite",
        "VGM",
        "TCMH",
        "CCMH",
        "Plaquettes",
        "VMP",
    ]
    cell = df.loc["TCMH", "Valeur"]
    assert isinstance(cell, Span)
    assert cell.text == "31.6"
