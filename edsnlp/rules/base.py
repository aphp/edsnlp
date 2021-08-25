from itertools import chain
from typing import List, Optional, Tuple

import mlconjug3
import pandas as pd
from spacy.tokens import Doc, Span

from spacy.util import filter_spans

if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)


class BaseComponent(object):
    """
    Base component that contains the logic for :

    - boundaries selections
    - match filtering
    - verbs conjugation
    """

    split_on_punctuation = True

    @staticmethod
    def _filter_matches(matches: List[Span]) -> List[Span]:
        """
        Filter matches to remove duplicates and inclusions.

        Arguments
        ---------
        matches: List of matches (spans).

        Returns
        -------
        filtered_matches: List of filtered matches.
        """

        return filter_spans(matches)

    def _boundaries(
        self, doc: Doc, terminations: Optional[List[Span]] = None
    ) -> List[Tuple[int, int]]:
        """
        Create sub sentences based sentences and terminations found in text.

        Parameters
        ----------
        doc:
            spaCy Doc object
        terminations:
            List of tuples with (match_id, start, end)

        Returns
        -------
        boundaries:
            List of tuples with (start, end) of spans
        """

        if terminations is None:
            terminations = []

        sent_starts = [sent.start for sent in doc.sents]
        termination_starts = [t.start for t in terminations]

        if self.split_on_punctuation:
            punctuations = [t.i for t in doc if t.is_punct and "-" not in t.text]
        else:
            punctuations = []

        starts = sent_starts + termination_starts + punctuations + [len(doc)]

        # Remove duplicates
        starts = list(set(starts))

        # Sort starts
        starts.sort()

        boundaries = [(start, end) for start, end in zip(starts[:-1], starts[1:])]

        return boundaries

    @staticmethod
    def _conjugate(verbs: List[str]) -> pd.DataFrame:
        """
        Create a list of conjugated verbs at all tenses from a list of infinitve verbs.

        Parameters
        ----------
        verbs:
            List of infinitive verbs to conjugate

        Returns
        ----------
        conjugated_verbs:
            Dataframe of conjugated verbs at all tenses
        """

        default_conjugator = mlconjug3.Conjugator(language="fr")

        conjugated_verbs = pd.DataFrame(
            columns=["infinitif", "mode", "temps", "personne", "variant"]
        )

        for verb in verbs:
            # Retrieve all conjugations
            conjugated_verb = default_conjugator.conjugate(verb)
            all_conjugated_forms = conjugated_verb.iterate()

            # Instantiate a dataframe with the retrieved conjugations
            df_verb = pd.DataFrame(
                all_conjugated_forms, columns=["mode", "temps", "personne", "variant"]
            )
            df_verb.insert(0, "infinitif", verb)

            # Manipulate the infinitive form
            df_verb.loc[df_verb["temps"] == "Infinitif Présent", "variant"] = verb
            df_verb.loc[df_verb["temps"] == "Infinitif Présent", "personne"] = None

            # Manipulate the present participle
            part_present = df_verb.loc[
                df_verb["temps"] == "Participe Présent", "personne"
            ]
            df_verb.loc[
                df_verb["temps"] == "Participe Présent", "variant"
            ] = part_present
            df_verb.loc[df_verb["temps"] == "Participe Présent", "personne"] = None

            conjugated_verbs = conjugated_verbs.append(df_verb)

        conjugated_verbs = conjugated_verbs.dropna()

        return conjugated_verbs
