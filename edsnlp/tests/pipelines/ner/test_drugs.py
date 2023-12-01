def test_drugs(blank_nlp):
    blank_nlp.add_pipe("eds.normalizer")
    blank_nlp.add_pipe("eds.drugs")

    text = "Traitement habituel: Kardégic, cardensiel (bisoprolol), glucophage, lasilix"
    doc = blank_nlp(text)
    drugs_expected = [
        ("Kardégic", "B01AC06"),
        ("cardensiel", "C07AB07"),
        ("bisoprolol", "C07AB07"),
        ("glucophage", "A10BA02"),
        ("lasilix", "C03CA01"),
    ]
    drugs_detected = [(x.text, x.kb_id_) for x in doc.ents]
    assert drugs_detected == drugs_expected
