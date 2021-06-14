"""
These section titles were extracted from a work performed by Ivan Lerner at AP-HP.
It supplied a number of documents annotated for section titles.

See sections/section-dataset notebook for detail.
"""

motif = [
    "motif d'hospitalisation",
    "motif de l'hospitalisation :",
    "motif",
    "motif de l'hospitalisation",
]

habitus = [
    "mode de vie",
    "mode de vie :",
    "habitus",
    "mode de vie - scolarite",
    "situation sociale, mode de vie",
    "contexte familial et social",
]

antecedents = [
    "antecedents",
    "antecedents personnels",
    "antecedents medicaux et chirurgicaux",
    "antecedents :",
]

traitements_entree = ["traitement a l'entree", "traitements a l'entree"]

histoire_de_la_maladie = [
    "histoire de la maladie",
    "histoire de la maladie actuelle",
    "histoire de la maladie - explorations",
    "histoire recente de la maladie",
]

examens_entree = ["examen clinique a l'entree", "examen clinique a l'entree :"]

examens_complementaires = [
    "examens complementaires realises pendant le sejour",
    "examens complementaires",
    "examen(s) complementaire(s) :",
    "examens complementaires a l'entree",
    "examens complementaires :",
    "examens para-cliniques",
]

evolution = ["evolution", "evolution depuis la derniere consultation"]

traitements_sortie = ["traitement de sortie", "traitement de sortie :"]

conclusion = [
    "au total",
    "conclusion de sortie",
    "synthese medicale/conclusion",
    "conclusion",
    "conclusion :",
    "synthese",
    "syntese medicale / conclusion",
]

facteurs_de_risques = ["facteurs de risques", "facteurs de risque"]

donnees_biometriques_entree = [
    "parametres vitaux et donnees biometriques a l'entree",
    "donnees biometriques et parametres vitaux a l'entree",
]

projet = ["projet diagnostique et therapeutique"]

examens = ["examen clinique", "examen clinique :"]

vaccination = ["vaccinations"]

planification = ["planification des soins"]

traitements_actuels = ["traitement actuel :"]

histoire_poids = ["histoire du poids :"]

activite_physique = ["enquete activite physique :"]

diagnostic = ["diagnostic", "diagnostics :", "diagnostics", "diagnostic retenu"]

actes = [
    "intervention",
    "actes realises :",
    "intervention(s) - acte(s) realise(s) :",
    "actes realises",
]

rappel_clinique = ["rappel clinique"]

description = ["description detaillee"]

histoire = ["histoire recente", "histoire de la maladie", "histoire de la maladie :"]

hemopathie = ["hemopathie"]

consignes = ["consignes a la sortie", "conduite a tenir :"]

suites = ["suites operatoires :"]

prescriptions = [
    "prescriptions medicales de sortie :",
    "prescriptions de sortie :",
    "prescriptions medicales de sortie",
]

indications = ["indication de l'acte"]

resultats = ["resultat de la coronarographie", "resultats d'examens"]

observations = ["observation"]

prise_en_charge = ["prise en charge"]

scores_entree = ["scores a l'entree"]

scores_sortie = ["scores a la sortie"]

resume = ["resume clinique", "resume"]

plan = ["planification des soins / suites a donner"]

entree = ["entree"]

traitements = ["traitement", "traitement en cours"]

suivi = ["suivi"]

antecedents_familiaux = ["antecedents familiaux"]

grossesse = ["grossesse - periode neonatale"]

conclusion_entree = ["conclusion a l'entree"]

plan_initial = ["attitude therapeutique initiale"]

depistages = ["depistages"]

destination = ["destination de sortie"]

pose_catheter = ["pose de catheter central"]

statut_sortie = ["statut fonctionnel de sortie"]

correspondants = ["correspondants"]

soins = ["soins infirmiers"]

sections = dict(
    motif=motif,
    habitus=habitus,
    antecedents=antecedents + antecedents_familiaux,
    traitement=traitements + traitements_actuels + traitements_entree + traitements_sortie,
    examens=examens + examens_entree + examens_complementaires,
    conclusion=conclusion + conclusion_entree,
    plan=plan + plan_initial + planification,
    resultats=resultats + scores_entree + scores_sortie,
    evolution=evolution,
    histoire=histoire + histoire_poids + histoire_de_la_maladie,
    diagnostic=diagnostic,
    prescriptions=prescriptions,
)
