from typing import List, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.matcher.matcher import GenericMatcher
from edsnlp.pipelines.misc.dates.factory import DEFAULT_CONFIG, DatesMatcher

from ...base import SpanSetterArg
from . import patterns as consult_regex


class ConsultationDatesMatcher(GenericMatcher):
    '''
    The `eds.consultation-dates` matcher consists of two main parts:

    - A **matcher** which finds mentions of _consultation events_ (more details below)
    - A **date parser** (see the corresponding pipe) that links a date to those events

    Examples
    --------
    !!! note

        The matcher has been built to run on **consultation notes**
        (`CR-CONS` at APHP), so please filter accordingly before proceeding.

    ```python
    import spacy

    # HIHIHI
    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe(
        "eds.normalizer",
        config=dict(
            lowercase=True,
            accents=True,
            quotes=True,
            pollution=False,
        ),
    )
    nlp.add_pipe("eds.consultation_dates")

    text = """
    XXX
    Objet : Compte-Rendu de Consultation du 03/10/2018.
    XXX
    """

    doc = nlp(text)

    doc.spans["consultation_dates"]
    # Out: [Consultation du 03/10/2018]

    doc.spans["consultation_dates"][0]._.consultation_date.to_datetime()
    # Out: DateTime(2018, 10, 3, 0, 0, 0, tzinfo=Timezone('Europe/Paris'))
    ```

    Extensions
    ----------
    The `eds.consultation_dates` pipeline declares one extension on the `Span` object:
    the `consultation_date` attribute, which is a Python `datetime` object.

    Parameters
    ----------
    nlp : Language
        Language pipeline object
    consultation_mention : Union[List[str], bool]
        List of RegEx for consultation mentions.

        - If `type==list`: Overrides the default list
        - If `type==bool`: Uses the default list of True, disable if False

        This list contains terms directly referring to consultations, such as
        "_Consultation du..._" or "_Compte rendu du..._". This list is the only one
        enabled by default since it is fairly precise and not error-prone.
    town_mention : Union[List[str], bool]
        List of RegEx for all AP-HP hospitals' towns mentions.

        - If `type==list`: Overrides the default list
        - If `type==bool`: Uses the default list of True, disable if False

        This list contains the towns of each AP-HP's hospital. Its goal is to fetch
        dates mentioned as "_Paris, le 13 décembre 2015_". It has a high recall but
        poor precision, since those dates can often be dates of letter redaction
        instead of consultation dates.
    document_date_mention : Union[List[str], bool]
        List of RegEx for document date.

        - If `type==list`: Overrides the default list
        - If `type==bool`: Uses the default list of True, disable if False

        This list contains expressions mentioning the date of creation/edition of a
        document, such as "_Date du rapport: 13/12/2015_" or "_Signé le 13/12/2015_".
        Like `town_mention` patterns, it has a high recall but is prone to errors since
        document date and consultation date aren't necessary similar.

    Authors and citation
    --------------------
    The `eds.consultation_dates` pipeline was developed by AP-HP's Data Science team.
    '''

    def __init__(
        self,
        nlp: Language,
        name: str = "eds.consultation_dates",
        *,
        consultation_mention: Union[List[str], bool] = True,
        town_mention: Union[List[str], bool] = False,
        document_date_mention: Union[List[str], bool] = False,
        attr: str = "NORM",
        ignore_excluded: bool = False,
        ignore_space_tokens: bool = False,
        label: str = "consultation_date",
        span_setter: SpanSetterArg = {"ents": True, "consultation_dates": True},
    ):
        logger.warning("This pipeline is still in beta")
        logger.warning(
            "This pipeline should ONLY be used on notes "
            "where `note_class_source_value == 'CR-CONS'`"
        )
        logger.warning(
            """This pipeline requires to use the normalizer pipeline with:
        lowercase=True,
        accents=True,
        quotes=True"""
        )

        if not (nlp.has_pipe("dates") and nlp.get_pipe("dates").on_ents_only is False):
            self.date_matcher = DatesMatcher(nlp, **DEFAULT_CONFIG)

        else:
            self.date_matcher = None

        if not consultation_mention:
            consultation_mention = []
        elif consultation_mention is True:
            consultation_mention = consult_regex.consultation_mention

        if not document_date_mention:
            document_date_mention = []
        elif document_date_mention is True:
            document_date_mention = consult_regex.document_date_mention

        if not town_mention:
            town_mention = []
        elif town_mention is True:
            town_mention = consult_regex.town_mention

        regex = dict(
            consultation_mention=consultation_mention,
            town_mention=town_mention,
            document_date_mention=document_date_mention,
        )
        self.label = label

        super().__init__(
            nlp=nlp,
            name=name,
            regex=regex,
            terms=dict(),
            attr=attr,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
            term_matcher="exact",
            term_matcher_config=dict(),
            span_setter=span_setter,
        )

        self.set_extensions()

    def set_extensions(self) -> None:
        super().set_extensions()
        if not Span.has_extension(self.label):
            Span.set_extension(self.label, default=None)

    def process(self, doc: Doc) -> List[Span]:
        """
        Finds entities

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        doc: Doc
            spaCy Doc object with additional
            `doc.spans['consultation_dates]` `SpanGroup`
        """

        matches = list(super().process(doc))

        self.date_matcher.span_getter = lambda d: [m.sent for m in matches]
        dates = [s for s in self.date_matcher.process(doc) if s.label_ == "date"]
        self.date_matcher.span_getter = None

        for match in matches:
            # Looking for a date
            # - In the same sentence
            # - Not less than 10 tokens AFTER the consultation mention
            matching_dates = [
                date
                for date in dates
                if (
                    (match.sent == date.sent)
                    and (date.start > match.start)
                    and (date.start - match.end <= 10)
                )
            ]

            if matching_dates:
                # We keep the first mention of a date
                kept_date = min(matching_dates, key=lambda d: d.start)
                span = doc[match.start : kept_date.end]
                span.label_ = self.label
                span._.consultation_date = kept_date._.date

                yield span
