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


"""


def test_tables(blank_nlp):
    blank_nlp.add_pipe("eds.normalizer")
    blank_nlp.add_pipe("eds.tables")

    doc = blank_nlp(TEXT)

    assert len(doc.spans["tables"]) == 1

    span = doc.spans["tables"][0]
    df = span._.to_pd_table()
    assert df.iloc[5, 0] == "TCMH "
