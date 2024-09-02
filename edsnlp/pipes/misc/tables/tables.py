import re
from typing import Dict, Optional, Union

import pandas as pd
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.misc.tables import patterns
from edsnlp.utils.typing import AsList


class TablesMatcher(BaseComponent):
    '''
    The `eds.tables` matcher detects tables in a documents.

    Examples
    --------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.normalizer())
    nlp.add_pipe(eds.tables())

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
    df = table._.to_pd_table(
        as_spans=False,  # set True to set the table cells as spans instead of strings
        header=False,  # set True to use the first row as header
        index=False,  # set True to use the first column as index
    )
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
    nlp : PipelineProtocol
        Pipeline object
    name: str
        Name of the component.
    tables_pattern : Optional[Dict[str, str]]
        The regex pattern to identify tables.
        The key of dictionary should be `tables`
    sep_pattern : Optional[str]
        The regex pattern to identify the separator pattern.
        Used when calling `to_pd_table`.
    min_rows : Optional[int]
        Only tables with more then `min_rows` lines will be detected.
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
        nlp: PipelineProtocol,
        name: Optional[str] = "tables",
        *,
        tables_pattern: Optional[AsList[str]] = None,
        sep_pattern: Optional[AsList[str]] = None,
        min_rows: int = 2,
        attr: Union[Dict[str, str], str] = "TEXT",
        ignore_excluded: bool = True,
    ):
        super().__init__(nlp, name)
        if tables_pattern is None:
            tables_pattern = patterns.regex_template

        if sep_pattern is None:
            sep_pattern = patterns.sep

        self.regex_matcher = RegexMatcher(attr=attr, ignore_excluded=ignore_excluded)
        self.regex_matcher.add(
            "table",
            list(
                dict.fromkeys(
                    template.format(sep=re.escape(sep), n=re.escape(str(min_rows)))
                    for sep in sep_pattern
                    for template in tables_pattern
                )
            ),
        )

        self.term_matcher = EDSPhraseMatcher(
            nlp.vocab, attr=attr, ignore_excluded=ignore_excluded
        )
        self.term_matcher.build_patterns(
            nlp,
            {
                "eol_pattern": "\n",
                "sep_pattern": sep_pattern,
            },
        )

        if not Span.has_extension("to_pd_table"):
            Span.set_extension("to_pd_table", method=self.to_pd_table)

    @classmethod
    def set_extensions(cls) -> None:
        """
        Set extensions for the tables pipeline.
        """

        if not Span.has_extension("table"):
            Span.set_extension("table", default=None)

    def get_table(self, table):
        """
        Convert spans of tables to dictionaries
        Parameters
        ----------
        table : Span

        Returns
        -------
        List[Span]
        """

        # We store each row in a list and store each of hese lists
        # in processed_table for post processing
        # considering the self.col_names and self.row_names var
        processed_table = []
        delimiters = [
            delimiter
            for delimiter in self.term_matcher(table, as_spans=True)
            if delimiter.start >= table.start and delimiter.end <= table.end
        ]

        last = table.start
        row = []
        # Parse the table to match each cell thanks to delimiters
        for delimiter in delimiters:
            row.append(table[last - table.start : delimiter.start - table.start])
            last = delimiter.end

            # End the actual row if there is an end of line
            if delimiter.label_ == "eol_pattern":
                processed_table.append(row)
                row = []

        # Remove first or last column in case the separator pattern is
        # also used in the raw table to draw the outlines
        max_len = max(len(row) for row in processed_table)
        if all(row[0].start == row[0].end for row in processed_table):
            processed_table = [row[1:] for row in processed_table]
        if all(
            row[-1].start == row[-1].end
            for row in processed_table
            if len(row) == max_len
        ):
            processed_table = [row[:-1] for row in processed_table]

        return processed_table

    def __call__(self, doc: Doc) -> Doc:
        """
        Find spans that contain tables

        Parameters
        ----------
        doc : Doc

        Returns
        -------
        Doc
        """
        matches = list(self.regex_matcher(doc, as_spans=True))
        doc.spans["tables"] = matches
        return doc

    def to_pd_table(
        self,
        span,
        as_spans=False,
        header: bool = False,
        index: bool = False,
    ) -> pd.DataFrame:
        """
        Return pandas DataFrame

        Parameters
        ----------
        span : Span
            The span containing the table
        as_spans : bool
            Whether to return the table cells as spans
        header : bool
            Whether the table has a header
        index : bool
            Whether the table has an index
        """
        table = self.get_table(span)
        if not as_spans:
            table = [[str(cell) for cell in data] for data in table]

        table = pd.DataFrame.from_records(table)
        if header:
            table.columns = [str(k) for k in table.iloc[0]]
            table = table[1:]
        if index:
            table.index = [str(k) for k in table.iloc[:, 0]]
            table = table.iloc[:, 1:]
        return table
