"""
These section titles were extracted from a work performed by Ivan Lerner at AP-HP.
It supplied a number of documents annotated for section titles.

The section titles were reviewed by Gilles Chatellier, who gave meaningful insights.

See sections/section-dataset notebook for detail.
"""

allergies = [r"allergies"]

antecedents = [
    r"antecedents",
    r"antecedents medicaux et chirurgicaux",
    r"antecedents personnels",
    r"antecedents medicaux",
    r"antecedents chirurgicaux",
    r"atcd",
]

antecedents_familiaux = [r"antecedents familiaux"]

traitements_entree = [
    r"attitude therapeutique initiale",
    r"traitement a l'entree",
    r"traitement actuel",
    r"traitement en cours",
    r"traitements a l'entree",
]

conclusion = [
    r"au total",
    r"conclusion",
    r"conclusion de sortie",
    r"syntese medicale / conclusion",
    r"synthese",
    r"synthese medicale",
    r"synthese medicale/conclusion",
    r"conclusion medicale",
]

conclusion_entree = [r"conclusion a l'entree"]

habitus = [
    r"contexte familial et social",
    r"habitus",
    r"mode de vie",
    r"mode de vie - scolarite",
    r"situation sociale, mode de vie",
]

correspondants = [r"correspondants"]

diagnostic = [r"diagnostic retenu"]

donnees_biometriques_entree = [
    r"donnees biometriques et parametres vitaux a l'entree",
    r"parametres vitaux et donnees biometriques a l'entree",
]

examens = [r"examen clinique", r"examen clinique a l'entree"]

examens_complementaires = [
    r"examen(s) complementaire(s)",
    r"examens complementaires",
    r"examens complementaires a l'entree",
    r"examens complementaires realises a l'entree",
    r"examens complementaires realises pendant le sejour",
    r"examens para-cliniques",
    r"imagerie post-operatoire",
]

facteurs_de_risques = [r"facteurs de risque", r"facteurs de risques"]

histoire_de_la_maladie = [
    r"histoire de la maladie",
    r"histoire de la maladie - explorations",
    r"histoire de la maladie actuelle",
    r"histoire du poids",
    r"histoire recente",
    r"histoire recente de la maladie",
    r"rappel clinique",
    r"resume",
    r"resume clinique",
    r"resume clinique - histoire de la maladie",
    r"antecedents et histoire de la maladie",
]

actes = [r"intervention"]

motif = [
    r"motif",
    r"motif d'hospitalisation",
    r"motif de l'hospitalisation",
    r"motif medical",
]

prescriptions = [r"prescriptions de sortie", r"prescriptions medicales de sortie"]

traitements_sortie = [r"traitement de sortie"]

evolution = [r"evolution", r"evolution et examen clinique aux lits portes :"]

modalites_sortie = [r"modalites de sortie", r"devenir du patient"]

vaccinations = [r"vaccinations", r"vaccination"]

introduction = [
    r"compte.?rendu d'hospitalisation.{0,30}",
]

sections = {
    r"allergies": allergies,
    r"antécédents": antecedents,
    r"antécédents familiaux": antecedents_familiaux,
    r"traitements entrée": traitements_entree,
    r"conclusion": conclusion,
    r"conclusion entrée": conclusion_entree,
    r"habitus": habitus,
    r"correspondants": correspondants,
    r"diagnostic": diagnostic,
    r"données biométriques entrée": donnees_biometriques_entree,
    r"examens": examens,
    r"examens complémentaires": examens_complementaires,
    r"facteurs de risques": facteurs_de_risques,
    r"histoire de la maladie": histoire_de_la_maladie,
    r"actes": actes,
    r"motif": motif,
    r"prescriptions": prescriptions,
    r"traitements sortie": traitements_sortie,
    r"evolution": evolution,
    r"modalites sortie": modalites_sortie,
    r"vaccinations": vaccinations,
    r"introduction": introduction,
}
