from typing import Dict, Optional, Union

import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.misc.tables import patterns


class TablesMatcher:
    """
    Pipeline to identify the Tables.

    It adds the key `tables` to doc.spans.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    tables_pattern : Optional[str]
        The regex pattern to identify tables.
    sep_pattern : Optional[str]
        The regex pattern to identify separators
        in the detected tables
    col_names : Optional[bool]
        Whether the tables_pattern matches column names
    row_names : Optional[bool]
        Whether the table_pattern matches row names
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with
        the key 'term_attr'. We can also add a key for each regex.
    ignore_excluded : bool
        Whether to skip excluded tokens.
    """

    def __init__(
        self,
        nlp: Language,
        tables_pattern: Optional[str],
        sep_pattern: Optional[str],
        attr: Union[Dict[str, str], str],
        ignore_excluded: bool,
        col_names: Optional[bool] = False,
        row_names: Optional[bool] = False,
    ):

        if tables_pattern is None:
            tables_pattern = patterns.regex

        if sep_pattern is None:
            sep_pattern = patterns.sep

        self.regex_matcher = RegexMatcher(attr=attr, ignore_excluded=True)
        self.regex_matcher.add("table", [tables_pattern])

        self.term_matcher = EDSPhraseMatcher(nlp.vocab, attr=attr, ignore_excluded=True)
        self.term_matcher.build_patterns(
            nlp,
            {
                "eol_pattern": "\n",
                "sep_pattern": sep_pattern,
            },
        )

        self.col_names = col_names
        self.row_names = row_names

        if not Span.has_extension("to_pd_table"):
            Span.set_extension("to_pd_table", method=self.to_pd_table)

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:
        """
        Set extensions for the tables pipeline.
        """

        if not Span.has_extension("table"):
            Span.set_extension("table", default=None)

    def get_tables(self, matches):
        """
        Convert spans of tables to dictionnaries

        Parameters
        ----------
        matches : List[Span]

        Returns
        -------
        List[Span]
        """

        # Dictionnaries linked to each table
        # Has the following format :
        # List[Dict[Union[str, int], List[Span]]]
        #   - List of dictionnaries containing the tables. Keys are
        #   column names (str) if col_names is set to True, else row
        #   names (str) if row_names is set to True, else index of
        #   column (int)
        tables_list = []

        # Returned list
        tables = []

        # Iter through matches to consider each table individually
        for table in matches:
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
            if all(row[0].start == row[0].end for row in processed_table):
                processed_table = [row[1:] for row in processed_table]
            if all(row[-1].start == row[-1].end for row in processed_table):
                processed_table = [row[:-1] for row in processed_table]

            tables_list.append(processed_table)

        # Convert to dictionnaries according to self.col_names
        # and self.row_names
        if self.col_names:
            for table_index in range(len(tables_list)):
                tables_list[table_index] = {
                    tables_list[table_index][0][column_index].text: [
                        tables_list[table_index][row_index][column_index]
                        for row_index in range(1, len(tables_list[table_index]))
                    ]
                    for column_index in range(len(tables_list[table_index][0]))
                }
        elif self.row_names:
            for table_index in range(len(tables_list)):
                tables_list[table_index] = {
                    tables_list[table_index][row_index][0].text: [
                        tables_list[table_index][row_index][column_index]
                        for column_index in range(1, len(tables_list[table_index][0]))
                    ]
                    for row_index in range(len(tables_list[table_index]))
                }
        else:
            for table_index in range(len(tables_list)):
                tables_list[table_index] = {
                    column_index: [
                        tables_list[table_index][row_index][column_index]
                        for row_index in range(len(tables_list[table_index]))
                    ]
                    for column_index in range(len(tables_list[table_index][0]))
                }

        for i in range(len(matches)):
            ent = matches[i]
            ent._.table = tables_list[i]
            tables.append(ent)

        return tables

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
        tables = self.get_tables(matches)
        doc.spans["tables"] = tables

        return doc

    def to_pd_table(self, span, as_spans=True) -> pd.DataFrame:
        """
        Return pandas DataFrame
        """
        if as_spans:
            table = span._.table
        else:
            table = {
                key: [str(cell) for cell in data]
                for key, data in list(span._.table.items())
            }
        if self.row_names:
            return pd.DataFrame.from_dict(table, orient="index")
        else:
            return pd.DataFrame.from_dict(table)
