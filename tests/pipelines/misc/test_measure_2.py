from pytest import raises
from spacy.language import Language

from edsnlp.pipelines.misc.measures import Measures

text = (
    "Le patient est admis hier, fait 1m78 pour 76kg. "
    "Les deux nodules bénins sont larges de 1,2 et 2.4mm évoluant vers"
    "La tumeur fait 1 x 2cm."
    "Angle de 12h12min "
    "Reduction de 12°12. "
    "Reduction de 12 degres12 "
    "Reduction de 12,20 degrés "
    "température de 37.2°C "
    "12cc de liquide "
    "Prescrit 2x500mg de paracétamol"
    "Deux gouttes de bicarbonate de sodium ou 2 gouttes de bicarbonate de sodium."
    "Pris 4 cl de bla et 12dL de blabla "
    "4 cl 89 de bla et 12 dL 56 de blabla "
    "45.12 decilitre "
    "Les trois kystes mesurent 1m, 2dm et 3cm. "
    "3kcal per dose of 45kg/mm "
    "3,12kJ of energy "
    "3245joule45 "
    "3mj12 "
    "45.23 psi of CO2 "
    "blood pressure: 89mmHg "
    "pressure: 45 hPa56 "
    "pressure: 1bar7 "
)

def test_measures_angle(blank_nlp: Language):
    blank_nlp.add_pipe(
        "eds.measures",
        config=dict(
            measures=["eds.measures.angle",]
        ),
    )
    doc = blank_nlp(text)
    m1, m2, m3, m4 = doc.spans["measures"]
    assert str(m1._.value) == "12.2h"
    assert str(m2._.value) == "12.2degree"
    assert str(m3._.value) == "12.2degree"
    assert str(m4._.value) == "12.2degree"
    
def test_measures_energy(blank_nlp: Language):
    blank_nlp.add_pipe(
        "eds.measures",
        config=dict(
            measures=["eds.measures.energy",]
        ),
    )
    doc = blank_nlp(text)
    m1, m2, m3, m4 = doc.spans["measures"]
    assert str(m1._.value) == "3.0kcal"
    assert str(m2._.value) == "3.12kJ"
    assert str(m3._.value) == "3245.45J"
    assert str(m4._.value) == "3.12MJ"
    
    text = "1 kilocal equals 4184 Joules"
    doc = blank_nlp(text)
    m1, m2 = doc.spans["measures"]
    assert m1._.value.J == m2._.value.value

def test_measures_pressure(blank_nlp: Language):
    blank_nlp.add_pipe(
        "eds.measures",
        config=dict(
            measures=["eds.measures.pressure",]
        ),
    )
    doc = blank_nlp(text)
    m1, m2, m3, m4 = doc.spans["measures"]
    assert str(m1._.value) == "45.23psi"
    assert str(m2._.value) == "89.0mmHg"
    assert str(m3._.value) == "45.56hPa"
    assert str(m4._.value) == "1.7bar"
    
    text = "blood pressure: 89mmHg4 equals 1bar"
    doc = blank_nlp(text)
    m1, m2 = doc.spans["measures"]
    assert m1._.value.bar == m2._.value.value
