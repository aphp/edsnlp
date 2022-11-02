from typing import List

pseudo: List[str] = [
    "aucun doute",
    "même si",
    "pas de condition",
    "pas de doute",
    "sans aucun doute",
    "sans condition",
    "sans risque",
]

confirmation: List[str] = [
    "certain",
    "certaine",
    "certainement",
    "certaines",
    "certains",
    "confirmer",
    "évidemment",
    "évident",
    "évidente",
    "montrer que",
    "visiblement",
]

preceding: List[str] = [
    "à condition",
    "à la condition que",
    "à moins que",
    "au cas où",
    "conditionnellement",
    "doute",
    "en admettant que",
    "en cas",
    "en considérant que",
    "en supposant que",
    "éventuellement",
    "exploration",
    "faudrait",
    "hypothèse",
    "hypothèses",
    "idée de",
    "pas confirmer",
    "pas sûr",
    "pas sûre",
    "peut correspondre",
    "peut-être",
    "peuvent correspondre",
    "possibilité",
    "possible",
    "possiblement",
    "potentiel",
    "potentielle",
    "potentiellement",
    "potentielles",
    "potentiels",
    "prédisposant à",
    "probable",
    "probablement",
    "probables",
    "recherche d'",
    "recherche de",
    "recherche des",
    "risque",
    "sauf si",
    "selon",
    "si",
    "s'il",
    "soit",
    "sous condition",
    "sous réserve",
    "suspicion",
]

following: List[str] = [
    "?",
    "envisagé",
    "envisageable",
    "envisageables",
    "envisagées",
    "envisagés",
    "hypothétique",
    "hypothétiquement",
    "hypothétiques",
    "pas certain",
    "pas certaine",
    "pas clair",
    "pas claire",
    "pas confirmé",
    "pas confirmée",
    "pas confirmées",
    "pas confirmés",
    "pas évident",
    "pas évidente",
    "pas sûr",
    "pas sûre",
    "possibilité",
    "possible",
    "potentiel",
    "potentielle",
    "potentiels",
    "probable",
    "probables",
]

verbs_hyp: List[str] = [
    "douter",
    "envisager",
    "explorer",
    "s'apparenter",
    "sembler",
    "soupçonner",
    "suggérer",
    "suspecter",
]

verbs_eds: List[str] = [
    "abandonner",
    "abolir",
    "aborder",
    "accepter",
    "accidenter",
    "accompagnemer",
    "accompagner",
    "acoller",
    "acquérir",
    "activer",
    "actualiser",
    "adapter",
    "adhérer",
    "adjuver",
    "admettre",
    "administrer",
    "adopter",
    "adresser",
    "aggraver",
    "agir",
    "agréer",
    "aider",
    "aimer",
    "alcooliser",
    "alerter",
    "alimenter",
    "aller",
    "allonger",
    "alléger",
    "alterner",
    "altérer",
    "amender",
    "amener",
    "améliorer",
    "amyotrophier",
    "améliorer",
    "analyser",
    "anesthésier",
    "animer",
    "annexer",
    "annuler",
    "anonymiser",
    "anticiper",
    "anticoaguler",
    "apercevoir",
    "aplatir",
    "apparaître",
    "appareiller",
    "appeler",
    "appliquer",
    "apporter",
    "apprendre",
    "apprécier",
    "appuyer",
    "argumenter",
    "arquer",
    "arrêter",
    "arriver",
    "arrêter",
    "articuler",
    "aspirer",
    "asseoir",
    "assister",
    "associer",
    "assurer",
    "assécher",
    "attacher",
    "atteindre",
    "attendre",
    "attribuer",
    "augmenter",
    "autonomiser",
    "autoriser",
    "avaler",
    "avancer",
    "avertir",
    "avoir",
    "avérer",
    "aérer",
    "baisser",
    "ballonner",
    "blesser",
    "bloquer",
    "boire",
    "border",
    "brancher",
    "brûler",
    "bénéficier",
    "cadrer",
    "calcifier",
    "calculer",
    "calmer",
    "canaliser",
    "capter",
    "carencer",
    "casser",
    "centrer",
    "cerner",
    "certifier",
    "changer",
    "charger",
    "chevaucher",
    "choisir",
    "chronomoduler",
    "chuter",
    "cicatriser",
    "circoncire",
    "circuler",
    "classer",
    "codéiner",
    "coincer",
    "colorer",
    "combler",
    "commander",
    "commencer",
    "communiquer",
    "comparer",
    "compliquer",
    "compléter",
    "comporter",
    "comprendre",
    "comprimer",
    "concerner",
    "conclure",
    "condamner",
    "conditionner",
    "conduire",
    "confiner",
    "confirmer",
    "confronter",
    "congeler",
    "conjoindre",
    "conjuguer",
    "connaître",
    "connecter",
    "conseiller",
    "conserver",
    "considérer",
    "consommer",
    "constater",
    "constituer",
    "consulter",
    "contacter",
    "contaminer",
    "contenir",
    "contentionner",
    "continuer",
    "contracter",
    "contrarier",
    "contribuer",
    "contrôler",
    "convaincre",
    "convenir",
    "convier",
    "convoquer",
    "copier",
    "correspondre",
    "corriger",
    "corréler",
    "coucher",
    "coupler",
    "couvrir",
    "crapotter",
    "creuser",
    "croire",
    "croiser",
    "créer",
    "crémer",
    "crépiter",
    "cumuler",
    "curariser",
    "céder",
    "dater",
    "demander",
    "demeurer",
    "destiner",
    "devenir",
    "devoir",
    "diagnostiquer",
    "dialyser",
    "dicter",
    "diffuser",
    "différencier",
    "différer",
    "digérer",
    "dilater",
    "diluer",
    "diminuer",
    "diner",
    "dire",
    "diriger",
    "discuter",
    "disparaître",
    "disposer",
    "dissocier",
    "disséminer",
    "disséquer",
    "distendre",
    "distinguer",
    "divorcer",
    "documenter",
    "donner",
    "dorer",
    "doser",
    "doubler",
    "durer",
    "dyaliser",
    "dyspner",
    "débuter",
    "décaler",
    "déceler",
    "décider",
    "déclarer",
    "déclencher",
    "découvrir",
    "décrire",
    "décroître",
    "décurariser",
    "décéder",
    "dédier",
    "définir",
    "dégrader",
    "délivrer",
    "dépasser",
    "dépendre",
    "déplacer",
    "dépolir",
    "déposer",
    "dériver",
    "dérouler",
    "désappareiller",
    "désigner",
    "désinfecter",
    "désorienter",
    "détecter",
    "déterminer",
    "détruire",
    "développer",
    "dévouer",
    "dîner",
    "écraser",
    "effacer",
    "effectuer",
    "effondrer",
    "emboliser",
    "emmener",
    "empêcher",
    "encadrer",
    "encourager",
    "endormir",
    "endurer",
    "enlever",
    "enregistrer",
    "entamer",
    "entendre",
    "entourer",
    "entraîner",
    "entreprendre",
    "entrer",
    "envahir",
    "envisager",
    "envoyer",
    "espérer",
    "essayer",
    "estimer",
    "être",
    "examiner",
    "excentrer",
    "exciser",
    "exclure",
    "expirer",
    "expliquer",
    "explorer",
    "exposer",
    "exprimer",
    "extérioriser",
    "exécuter",
    "faciliter",
    "faire",
    "fatiguer",
    "favoriser",
    "faxer",
    "fermer",
    "figurer",
    "fixer",
    "focaliser",
    "foncer",
    "former",
    "fournir",
    "fractionner",
    "fragmenter",
    "fuiter",
    "fusionner",
    "garder",
    "graver",
    "guider",
    "gérer",
    "gêner",
    "honorer",
    "hopsitaliser",
    "hospitaliser",
    "hydrater",
    "hyperartérialiser",
    "hyperfixer",
    "hypertrophier",
    "hésiter",
    "identifier",
    "illustrer",
    "immuniser",
    "impacter",
    "implanter",
    "impliquer",
    "importer",
    "imposer",
    "impregner",
    "imprimer",
    "inclure",
    "indifferencier",
    "indiquer",
    "infecter",
    "infertiliser",
    "infiltrer",
    "informer",
    "inhaler",
    "initier",
    "injecter",
    "inscrire",
    "insister",
    "installer",
    "interdire",
    "interpréter",
    "interrompre",
    "intervenir",
    "intituler",
    "introduire",
    "intéragir",
    "inverser",
    "inviter",
    "ioder",
    "ioniser",
    "irradier",
    "itérativer",
    "joindre",
    "juger",
    "justifier",
    "laisser",
    "laminer",
    "lancer",
    "latéraliser",
    "laver",
    "lever",
    "lier",
    "ligaturer",
    "limiter",
    "lire",
    "localiser",
    "loger",
    "louper",
    "luire",
    "lutter",
    "lyricer",
    "lyser",
    "maculer",
    "macérer",
    "maintenir",
    "majorer",
    "malaiser",
    "manger",
    "manifester",
    "manipuler",
    "manquer",
    "marcher",
    "marier",
    "marmoner",
    "marquer",
    "masquer",
    "masser",
    "mater",
    "mener",
    "mesurer",
    "meteoriser",
    "mettre",
    "mitiger",
    "modifier",
    "moduler",
    "modérer",
    "monter",
    "montrer",
    "motiver",
    "moucheter",
    "mouler",
    "mourir",
    "multiopéréer",
    "munir",
    "muter",
    "médicaliser",
    "météoriser",
    "naître",
    "normaliser",
    "noter",
    "nuire",
    "numériser",
    "nécessiter",
    "négativer",
    "objectiver",
    "observer",
    "obstruer",
    "obtenir",
    "occasionner",
    "occuper",
    "opposer",
    "opérer",
    "organiser",
    "orienter",
    "ouvrir",
    "palper",
    "parasiter",
    "paraître",
    "parcourir",
    "parer",
    "paresthésier",
    "parfaire",
    "partager",
    "partir",
    "parvenir",
    "passer",
    "penser",
    "percevoir",
    "perdre",
    "perforer",
    "permettre",
    "persister",
    "personnaliser",
    "peser",
    "pigmenter",
    "piloter",
    "placer",
    "plaindre",
    "planifier",
    "plier",
    "plonger",
    "porter",
    "poser",
    "positionner",
    "posséder",
    "poursuivre",
    "pousser",
    "pouvoir",
    "pratiquer",
    "preciser",
    "prendre",
    "prescrire",
    "prier",
    "produire",
    "programmer",
    "prolonger",
    "prononcer",
    "proposer",
    "prouver",
    "provoquer",
    "préciser",
    "précéder",
    "prédominer",
    "préexister",
    "préférer",
    "prélever",
    "préparer",
    "présenter",
    "préserver",
    "prévenir",
    "prévoir",
    "puruler",
    "pénétrer",
    "radiofréquencer",
    "ralentir",
    "ramener",
    "rappeler",
    "rapporter",
    "rapprocher",
    "rassurer",
    "rattacher",
    "rattraper",
    "realiser",
    "recenser",
    "recevoir",
    "rechercher",
    "recommander",
    "reconnaître",
    "reconsulter",
    "recontacter",
    "recontrôler",
    "reconvoquer",
    "recouvrir",
    "recueillir",
    "recuperer",
    "redescendre",
    "rediscuter",
    "refaire",
    "refouler",
    "refuser",
    "regarder",
    "rehausser",
    "relancer",
    "relayer",
    "relever",
    "relire",
    "relâcher",
    "remanier",
    "remarquer",
    "remercier",
    "remettre",
    "remonter",
    "remplacer",
    "remplir",
    "rencontrer",
    "rendormir",
    "rendre",
    "renfermer",
    "renforcer",
    "renouveler",
    "renseigner",
    "rentrer",
    "reparler",
    "repasser",
    "reporter",
    "reprendre",
    "represcrire",
    "reproduire",
    "reprogrammer",
    "représenter",
    "repérer",
    "requérir",
    "respecter",
    "ressembler",
    "ressentir",
    "rester",
    "restreindre",
    "retarder",
    "retenir",
    "retirer",
    "retrouver",
    "revasculariser",
    "revenir",
    "reverticaliser",
    "revoir",
    "rompre",
    "rouler",
    "réadapter",
    "réadmettre",
    "réadresser",
    "réaliser",
    "récidiver",
    "récupérer",
    "rédiger",
    "réduire",
    "réessayer",
    "réexpliquer",
    "référer",
    "régler",
    "régresser",
    "réhausser",
    "réopérer",
    "répartir",
    "répondre",
    "répéter",
    "réserver",
    "résorber",
    "résoudre",
    "réséquer",
    "réveiller",
    "révéler",
    "réévaluer",
    "rêver",
    "sacrer",
    "saisir",
    "satisfaire",
    "savoir",
    "scanner",
    "scolariser",
    "sembler",
    "sensibiliser",
    "sentir",
    "serrer",
    "servir",
    "sevrer",
    "signaler",
    "signer",
    "situer",
    "siéger",
    "soigner",
    "sommeiller",
    "sonder",
    "sortir",
    "souffler",
    "souhaiter",
    "soulager",
    "soussigner",
    "souvenir",
    "spécialiser",
    "stabiliser",
    "statuer",
    "stenter",
    "stopper",
    "stratifier",
    "subir",
    "substituer",
    "sucrer",
    "suggérer",
    "suivre",
    "supporter",
    "supprimer",
    "surajouter",
    "surmonter",
    "surveiller",
    "survenir",
    "suspecter",
    "suspendre",
    "suturer",
    "synchroniser",
    "systématiser",
    "sécréter",
    "sécuriser",
    "sédater",
    "séjourner",
    "séparer",
    "taire",
    "taper",
    "teinter",
    "tendre",
    "tenir",
    "tenter",
    "terminer",
    "tester",
    "thromboser",
    "tirer",
    "tiroir",
    "tissulaire",
    "titulariser",
    "tolérer",
    "tourner",
    "tracer",
    "trachéotomiser",
    "traduire",
    "traiter",
    "transcrire",
    "transférer",
    "transmettre",
    "transporter",
    "trasnfixer",
    "travailler",
    "tronquer",
    "trouver",
    "téléphoner",
    "ulcérer",
    "uriner",
    "utiliser",
    "vacciner",
    "valider",
    "valoir",
    "varier",
    "vasculariser",
    "venir",
    "verifier",
    "vieillir",
    "viser",
    "visualiser",
    "vivre",
    "voir",
    "vouloir",
    "vérifier",
    "ébaucher",
    "écarter",
    "échographier",
    "échoguider",
    "échoir",
    "échouer",
    "éclairer",
    "écraser",
    "élargir",
    "éliminer",
    "émousser",
    "épaissir",
    "épargner",
    "épuiser",
    "épurer",
    "équilibrer",
    "établir",
    "étager",
    "étendre",
    "étiqueter",
    "étrangler",
    "évaluer",
    "éviter",
    "évoluer",
    "évoquer",
    "être",
]
