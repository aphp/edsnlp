from typing import List, Union

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.matcher import GenericMatcher
from edsnlp.pipelines.misc.dates import Dates
from edsnlp.pipelines.misc.dates.factory import DEFAULT_CONFIG

from . import patterns as consult_regex


class ConsultationDates(GenericMatcher):
    """
    Class to extract consultation dates from "CR-CONS" documents.

    The pipeline populates the `#!python doc.spans['consultation_dates']` list.

    For each extraction `s` in this list, the corresponding date is available
    as `s._.consultation_date`.

    Parameters
    ----------
    nlp : Language
        Language pipeline object
    consultation_mention : Union[List[str], bool]
        List of RegEx for consultation mentions.

        - If `type==list`: Overrides the default list
        - If `type==bool`: Uses the default list of True, disable if False

    town_mention : Union[List[str], bool]
        List of RegEx for all AP-HP hospitals' towns mentions.

        - If `type==list`: Overrides the default list
        - If `type==bool`: Uses the default list of True, disable if False
    document_date_mention : Union[List[str], bool]
        List of RegEx for document date.

        - If `type==list`: Overrides the default list
        - If `type==bool`: Uses the default list of True, disable if False
    """

    def __init__(
        self,
        nlp: Language,
        consultation_mention: Union[List[str], bool],
        town_mention: Union[List[str], bool],
        document_date_mention: Union[List[str], bool],
        attr: str,
        **kwargs,
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

            config = dict(**DEFAULT_CONFIG)
            config["on_ents_only"] = "consultation_mentions"

            self.date_matcher = Dates(nlp, **config)

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

        super().__init__(
            nlp,
            regex=regex,
            terms=dict(),
            attr=attr,
            ignore_excluded=False,
            **kwargs,
        )

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:
        if not Span.has_extension("consultation_date"):
            Span.set_extension("consultation_date", default=None)

    def __call__(self, doc: Doc) -> Doc:
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

        ents = self.process(doc)

        doc.spans["consultation_mentions"] = ents
        doc.spans["consultation_dates"] = []

        if self.date_matcher is not None:
            doc = self.date_matcher(doc)

        for mention in ents:
            # Looking for a date
            # - In the same sentence
            # - Not less than 10 tokens AFTER the consultation mention
            matching_dates = [
                date
                for date in doc.spans["dates"]
                if (
                    (mention.sent == date.sent)
                    and (date.start > mention.start)
                    and (date.start - mention.end <= 10)
                )
            ]

            if matching_dates:
                # We keep the first mention of a date
                kept_date = min(matching_dates, key=lambda d: d.start)
                span = doc[mention.start : kept_date.end]
                span.label_ = mention.label_
                span._.consultation_date = kept_date._.date

                doc.spans["consultation_dates"].append(span)

        del doc.spans["consultation_mentions"]

        return doc
