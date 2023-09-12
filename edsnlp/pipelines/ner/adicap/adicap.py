"""`eds.adicap` pipeline"""
import re
from typing import List, Optional, Union

from spacy import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.utils.resources import get_adicap_dict

from . import patterns
from .models import AdicapCode


# noinspection SpellCheckingInspection
class AdicapMatcher(ContextualMatcher):
    """
    The `eds.adicap` pipeline component matches the ADICAP codes. It was developped to
    run on anapathology reports.

    !!! warning "Document type"

        It was developped to work on anapathology reports.
        We recommend also to use the `eds` language (`spacy.blank("eds")`)

    The compulsory characters of the ADICAP code are identified and decoded.
    These characters represent the following attributes:

    | Field [en]        | Field [fr]                    | Attribute       |
    |-------------------|-------------------------------|-----------------|
    | Sampling mode     | Mode de prelevement           | sampling_mode   |
    | Technic           | Type de technique             | technic         |
    | Organ and regions | Appareils, organes et régions | organ           |
    | Pathology         | Pathologie générale           | pathology       |
    | Pathology type    | Type de la pathologie         | pathology_type  |
    | Behaviour type    | Type de comportement          | behaviour_type  |


    The pathology field takes 4 different values corresponding to the 4 possible
    interpretations of the ADICAP code, which are : "PATHOLOGIE GÉNÉRALE NON TUMORALE",
    "PATHOLOGIE TUMORALE", "PATHOLOGIE PARTICULIERE DES ORGANES" and "CYTOPATHOLOGIE".

    Depending on the pathology value the behaviour type meaning changes, when the
    pathology is tumoral then it describes the malignancy of the tumor.

    For further details about the ADICAP code follow this [link](https://smt.esante.\
gouv.fr/wp-json/ans/terminologies/document?terminologyId=terminologie-adicap&file\
Name=cgts_sem_adicap_fiche-detaillee.pdf).

    Examples
    --------
    ```{ .python .no-check }
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.adicap")

    text = \"\"\"
    COMPTE RENDU D’EXAMEN

    Antériorité(s) :  NEANT


    Renseignements cliniques :
    Contexte d'exploration d'un carcinome canalaire infiltrant du quadrant supéro-
    externe du sein droit. La lésion biopsiée ce jour est située à 5,5 cm de la lésion
    du quadrant supéro-externe, à l'union des quadrants inférieurs.


    Macrobiopsie 10G sur une zone de prise de contraste focale à l'union des quadrants
    inférieurs du sein droit, mesurant 4 mm, classée ACR4

    14 fragments ont été communiqués fixés en formol (lame n° 1a et lame n° 1b) . Il
    n'y a pas eu d'échantillon congelé. Ces fragments ont été inclus en paraffine en
    totalité et coupés sur plusieurs niveaux.
    Histologiquement, il s'agit d'un parenchyme mammaire fibroadipeux parfois
    légèrement dystrophique avec quelques petits kystes. Il n'y a pas d'hyperplasie
    épithéliale, pas d'atypie, pas de prolifération tumorale. On note quelques
    suffusions hémorragiques focales.

    Conclusion :
    Légers remaniements dystrophiques à l'union des quadrants inférieurs du sein droit.
    Absence d'atypies ou de prolifération tumorale.

    Codification :   BHGS0040
    \"\"\"

    doc = nlp(text)

    doc.ents
    # Out: (BHGS0040,)

    ent = doc.ents[0]

    ent.label_
    # Out: adicap

    ent._.adicap.dict()
    # Out: {'code': 'BHGS0040',
    # 'sampling_mode': 'BIOPSIE CHIRURGICALE',
    # 'technic': 'HISTOLOGIE ET CYTOLOGIE PAR INCLUSION',
    # 'organ': "SEIN (ÉGALEMENT UTILISÉ CHEZ L'HOMME)",
    # 'pathology': 'PATHOLOGIE GÉNÉRALE NON TUMORALE',
    # 'pathology_type': 'ETAT SUBNORMAL - LESION MINEURE',
    # 'behaviour_type': 'CARACTERES GENERAUX'}
    ```

    Parameters
    ----------
    nlp : Optional[Language]
        The pipeline object
    name : str
        The name of the pipe
    pattern : Optional[Union[List[str], str]]
        The regex pattern to use for matching ADICAP codes
    prefix : Optional[Union[List[str], str]]
        The regex pattern to use for matching the prefix before ADICAP codes
    window : int
        Number of tokens to look for prefix. It will never go further the start of
        the sentence
    attr : str
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    label : str
        Label name to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Authors and citation
    --------------------
    The `eds.adicap` pipeline was developed by AP-HP's Data Science team.
    The codes were downloaded from the website of 'Agence du numérique en santé'
    ("Thésaurus de la codification ADICAP - Index raisonné des
    lésions", [@terminologie-adicap])
    """

    def __init__(
        self,
        nlp: Optional[Language],
        name: str = "eds.adicap",
        *,
        pattern: Union[List[str], str] = patterns.base_code,
        prefix: Union[List[str], str] = patterns.adicap_prefix,
        window: int = 500,
        attr: str = "TEXT",
        label: str = "adicap",
        span_setter: SpanSetterArg = {"ents": True, "adicap": True},
    ):
        adicap_pattern = dict(
            source="adicap",
            regex=prefix,
            regex_attr=attr,
            assign=[
                dict(
                    name="code",
                    regex=pattern,
                    window=window,
                    replace_entity=True,
                    reduce_mode=None,
                ),
            ],
        )

        super().__init__(
            nlp=nlp,
            name=name,
            label=label,
            attr=attr,
            patterns=adicap_pattern,
            ignore_excluded=False,
            regex_flags=0,
            alignment_mode="expand",
            include_assigned=False,
            assign_as_span=False,
            span_setter=span_setter,
        )

        self.decode_dict = get_adicap_dict()

        self.set_extensions()

    def set_extensions(self) -> None:
        super().set_extensions()
        if not Span.has_extension(self.label):
            Span.set_extension(self.label, default=None)

    def decode(self, code):
        code = re.sub("[^A-Za-z0-9 ]+", "", code)
        exploded = list(code)
        adicap = AdicapCode(
            code=code,
            sampling_mode=self.decode_dict["D1"]["codes"].get(exploded[0]),
            technic=self.decode_dict["D2"]["codes"].get(exploded[1]),
            organ=self.decode_dict["D3"]["codes"].get("".join(exploded[2:4])),
        )

        for d in ["D4", "D5", "D6", "D7"]:
            adicap_short = self.decode_dict[d]["codes"].get("".join(exploded[4:8]))
            adicap_long = self.decode_dict[d]["codes"].get("".join(exploded[2:8]))

            if (adicap_short is not None) | (adicap_long is not None):
                adicap.pathology = self.decode_dict[d]["label"]
                adicap.behaviour_type = self.decode_dict[d]["codes"].get(exploded[5])

                if adicap_short is not None:
                    adicap.pathology_type = adicap_short

                else:
                    adicap.pathology_type = adicap_long

        return adicap

    def process(self, doc: Doc) -> List[Span]:
        """
        Tags ADICAP mentions.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for ADICAP
        """
        for span in super().process(doc):
            span._.set(self.label, self.decode(span._.assigned["code"]))
            yield span
