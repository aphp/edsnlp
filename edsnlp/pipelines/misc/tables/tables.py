from io import StringIO
from typing import Dict, Optional, Union

import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.matcher.matcher import GenericMatcher
from edsnlp.pipelines.misc.tables import patterns
from edsnlp.utils.filter import get_spans


class TablesMatcher(GenericMatcher):
    '''
    The `eds.tables` matcher detects tables in a documents.

    Examples
    --------
    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.tables")

    text = """
    SERVICE
    MEDECINE INTENSIVE –
    REANIMATION
    Réanimation / Surveillance Continue
    Médicale

    COMPTE RENDU D'HOSPITALISATION du 05/06/2020 au 10/06/2020
    Madame DUPONT Marie, née le 16/05/1900, âgée de 20 ans, a été hospitalisée en
    réanimation du 05/06/1920 au 10/06/1920 pour intoxication médicamenteuse volontaire.

    Examens complémentaires
    Hématologie
    Numération
    Leucocytes ¦x10*9/L ¦4.97 ¦4.09-11
    Hématies ¦x10*12/L¦4.68 ¦4.53-5.79
    Hémoglobine ¦g/dL ¦14.8 ¦13.4-16.7
    Hématocrite ¦% ¦44.2 ¦39.2-48.6
    VGM ¦fL ¦94.4 + ¦79.6-94
    TCMH ¦pg ¦31.6 ¦27.3-32.8
    CCMH ¦g/dL ¦33.5 ¦32.4-36.3
    Plaquettes ¦x10*9/L ¦191 ¦172-398
    VMP ¦fL ¦11.5 + ¦7.4-10.8

    Sur le plan neurologique : Devant la persistance d'une confusion à distance de
    l'intoxication au
    ...

    2/2Pat : <NOM> <Prenom>|F |<date> | <ipp> |Intitulé RCP
    """

    doc = nlp(text)

    # A table span
    table = doc.spans["tables"][0]

    # Leucocytes ¦x10*9/L ¦4.97 ¦4.09-11
    # Hématies ¦x10*12/L¦4.68 ¦4.53-5.79
    # Hémoglobine ¦g/dL ¦14.8 ¦13.4-16.7
    # Hématocrite ¦% ¦44.2 ¦39.2-48.6
    # VGM ¦fL ¦94.4 + ¦79.6-94
    # TCMH ¦pg ¦31.6 ¦27.3-32.8
    # CCMH ¦g/dL ¦33.5 ¦32.4-36.3
    # Plaquettes ¦x10*9/L ¦191 ¦172-398
    # VMP ¦fL ¦11.5 + ¦7.4-10.8

    # Convert span to Pandas table
    df = table._.to_pd_table()
    type(df)
    # Out: pandas.core.frame.DataFrame
    ```
    The pandas DataFrame:

    |      | 0           | 1        | 2      | 3         |
    | ---: | :---------- | :------- | :----- | :-------- |
    |    0 | Leucocytes  | x10*9/L  | 4.97   | 4.09-11   |
    |    1 | Hématies    | x10*12/L | 4.68   | 4.53-5.79 |
    |    2 | Hémoglobine | g/dL     | 14.8   | 13.4-16.7 |
    |    3 | Hématocrite | %        | 44.2   | 39.2-48.6 |
    |    4 | VGM         | fL       | 94.4 + | 79.6-94   |
    |    5 | TCMH        | pg       | 31.6   | 27.3-32.8 |
    |    6 | CCMH        | g/dL     | 33.5   | 32.4-36.3 |
    |    7 | Plaquettes  | x10*9/L  | 191    | 172-398   |
    |    8 | VMP         | fL       | 11.5 + | 7.4-10.8  |

    Extensions
    ----------
    The `eds.tables` pipeline declares the `span._.to_pd_table()` Span extension.
    This function returns a parsed pandas version of the table.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    name: str
        Name of the component.
    tables_pattern : Optional[Dict[str, str]]
        The regex pattern to identify tables.
        The key of dictionary should be `tables`
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with
        the key 'term_attr'. We can also add a key for each regex.
    ignore_excluded : bool
        Whether to skip excluded tokens.

    Authors and citation
    --------------------
    The `eds.tables` pipeline was developed by AP-HP's Data Science team.
    '''

    def __init__(
        self,
        nlp: Language,
        name: str = "eds.tables",
        *,
        tables_pattern: Optional[Dict[str, str]] = None,
        sep_pattern: Optional[str] = None,
        attr: Union[Dict[str, str], str] = "TEXT",
        ignore_excluded: bool = True,
    ):
        if tables_pattern is None:
            self.tables_pattern = patterns.regex
        else:
            self.tables_pattern = tables_pattern

        if sep_pattern is None:
            self.sep = patterns.sep
        else:
            self.sep = sep_pattern

        super().__init__(
            nlp=nlp,
            name=name,
            terms=None,
            regex=self.tables_pattern,
            attr=attr,
            ignore_excluded=ignore_excluded,
        )

        if not Span.has_extension("to_pd_table"):
            Span.set_extension("to_pd_table", method=self.to_pd_table)

        self.set_extensions()

    def __call__(self, doc: Doc) -> Doc:
        """Find spans that contain tables

        Parameters
        ----------
        doc : Doc

        Returns
        -------
        Doc
        """
        matches = self.process(doc)
        tables = get_spans(matches, "tables")
        # parsed = self.parse(tables=tables)

        doc.spans["tables"] = tables

        return doc

    def to_pd_table(self, span) -> pd.DataFrame:
        table_str_io = StringIO(span.text)
        parsed = pd.read_csv(
            table_str_io,
            sep=self.sep,
            engine="python",
            header=None,
            on_bad_lines="skip",
        )
        return parsed
