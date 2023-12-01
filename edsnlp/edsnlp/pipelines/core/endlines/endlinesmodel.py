import pickle
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from numpy.lib.function_base import iterable
from pandas.api.types import CategoricalDtype
from pandas.core.groupby import DataFrameGroupBy
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from spacy.language import Language
from spacy.strings import StringStore
from spacy.tokens import Doc

from .functional import _convert_series_to_array


class EndLinesModel:
    """Model to classify if an end line is a real one or it should be a space.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    """

    def __init__(self, nlp: Language):
        self.nlp = nlp

    def _preprocess_data(self, corpus: Iterable[Doc]) -> pd.DataFrame:
        """
        Parameters
        ----------
        corpus : Iterable[Doc]
            Corpus of documents

        Returns
        -------
        pd.DataFrame
            Preprocessed data
        """
        # Extract the vocabulary
        string_store = self.nlp.vocab.strings

        # Iterate in the corpus and construct a dataframe
        train_data_list = []
        for i, doc in enumerate(corpus):
            train_data_list.append(self._get_attributes(doc, i))

        df = pd.concat(train_data_list)
        df.reset_index(inplace=True, drop=False)
        df.rename(columns={"ORTH": "A1", "index": "original_token_index"}, inplace=True)

        # Retrieve string representation of token_id and shape
        df["TEXT"] = df.A1.apply(self._get_string, string_store=string_store)
        df["SHAPE_"] = df.SHAPE.apply(self._get_string, string_store=string_store)

        # Convert new lines as an attribute instead of a row
        df = self._convert_line_to_attribute(df, expr="\n", col="END_LINE")
        df = self._convert_line_to_attribute(df, expr="\n\n", col="BLANK_LINE")
        df = df.loc[~(df.END_LINE | df.BLANK_LINE)]
        df = df.drop(columns="END_LINE")
        df = df.drop(columns="BLANK_LINE")
        df.rename(
            columns={"TEMP_END_LINE": "END_LINE", "TEMP_BLANK_LINE": "BLANK_LINE"},
            inplace=True,
        )

        # Construct A2 by shifting
        df = self._shift_col(df, "A1", "A2", direction="backward")

        # Compute A3 and A4
        df = self._compute_a3(df)
        df = self._shift_col(df, "A3", "A4", direction="backward")

        # SPACE is the class to predict. Set 1 if not an END_LINE
        df["SPACE"] = np.logical_not(df["END_LINE"]).astype("int")

        df[["END_LINE", "BLANK_LINE"]] = df[["END_LINE", "BLANK_LINE"]].fillna(
            True, inplace=False
        )

        # Assign a sentence id to each token
        df = df.groupby("DOC_ID").apply(self._retrieve_lines)
        df["SENTENCE_ID"] = df["SENTENCE_ID"].astype("int")

        # Compute B1 and B2
        df = self._compute_B(df)

        # Drop Tokens without info (last token of doc)
        df.dropna(subset=["A1", "A2", "A3", "A4"], inplace=True)

        # Export the vocabularies to be able to use the model with another corpus
        voc_a3a4 = self._create_vocabulary(df.A3_.cat.categories)
        voc_B2 = self._create_vocabulary(df.cv_bin.cat.categories)
        voc_B1 = self._create_vocabulary(df.l_norm_bin.cat.categories)

        vocabulary = {"A3A4": voc_a3a4, "B1": voc_B1, "B2": voc_B2}

        self.vocabulary = vocabulary

        return df

    def fit_and_predict(self, corpus: Iterable[Doc]) -> pd.DataFrame:
        """Fit the model and predict for the training data

        Parameters
        ----------
        corpus : Iterable[Doc]
            An iterable of Documents

        Returns
        -------
        pd.DataFrame
            one line by end_line prediction
        """

        # Preprocess data to have a pd DF
        df = self._preprocess_data(corpus)

        # Train and predict M1
        self._fit_M1(df.A1, df.A2, df.A3, df.A4, df.SPACE)
        outputs_M1 = self._predict_M1(
            df.A1,
            df.A2,
            df.A3,
            df.A4,
        )
        df["M1"] = outputs_M1["predictions"]
        df["M1_proba"] = outputs_M1["predictions_proba"]

        # Force Blank lines to 0
        df.loc[df.BLANK_LINE, "M1"] = 0

        # Train and predict M2
        df_endlines = df.loc[df.END_LINE]
        self._fit_M2(B1=df_endlines.B1, B2=df_endlines.B2, label=df_endlines.M1)
        outputs_M2 = self._predict_M2(B1=df_endlines.B1, B2=df_endlines.B2)

        df.loc[df.END_LINE, "M2"] = outputs_M2["predictions"]
        df.loc[df.END_LINE, "M2_proba"] = outputs_M2["predictions_proba"]

        df["M2"] = df["M2"].astype(
            pd.Int64Dtype()
        )  # cast to pd.Int64Dtype cause there are None values

        # M1M2
        df = df.loc[df.END_LINE]
        df["M1M2_lr"] = (df["M2_proba"] / (1 - df["M2_proba"])) * (
            df["M1_proba"] / (1 - df["M1_proba"])
        )
        df["M1M2"] = (df["M1M2_lr"] > 1).astype("int")

        # Force Blank lines to 0
        df.loc[df.BLANK_LINE, ["M2", "M1M2"]] = 0

        # Make binary col
        df["PREDICTED_END_LINE"] = np.logical_not(df["M1M2"].astype(bool))

        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use the model for inference

        The df should have the following columns:
        `["A1","A2","A3","A4","B1","B2","BLANK_LINE"]`

        Parameters
        ----------
        df : pd.DataFrame
            The df should have the following columns:
            `["A1","A2","A3","A4","B1","B2","BLANK_LINE"]`

        Returns
        -------
        pd.DataFrame
            The result is added to the column `PREDICTED_END_LINE`
        """

        df = self._convert_raw_data_to_codes(df)

        outputs_M1 = self._predict_M1(df.A1, df.A2, df._A3, df._A4)
        df["M1"] = outputs_M1["predictions"]
        df["M1_proba"] = outputs_M1["predictions_proba"]

        outputs_M2 = self._predict_M2(B1=df._B1, B2=df._B2)
        df["M2"] = outputs_M2["predictions"]
        df["M2_proba"] = outputs_M2["predictions_proba"]
        df["M2"] = df["M2"].astype(
            pd.Int64Dtype()
        )  # cast to pd.Int64Dtype cause there are None values

        # M1M2
        df["M1M2_lr"] = (df["M2_proba"] / (1 - df["M2_proba"])) * (
            df["M1_proba"] / (1 - df["M1_proba"])
        )
        df["M1M2"] = (df["M1M2_lr"] > 1).astype("int")

        # Force Blank lines to 0
        df.loc[
            df.BLANK_LINE,
            [
                "M1M2",
            ],
        ] = 0

        # Make binary col
        df["PREDICTED_END_LINE"] = np.logical_not(df["M1M2"].astype(bool))

        return df

    def save(self, path="base_model.pkl"):
        """Save a pickle of the model. It could be read by the pipeline later.

        Parameters
        ----------
        path : str, optional
            path to file .pkl, by default `base_model.pkl`
        """
        with open(path, "wb") as outp:
            del self.nlp
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def _convert_A(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
        col : str
            column to translate

        Returns
        -------
        pd.DataFrame
        """
        cat_type_A = CategoricalDtype(
            categories=self.vocabulary["A3A4"].keys(), ordered=True
        )
        new_col = "_" + col
        df[new_col] = df[col].astype(cat_type_A)
        df[new_col] = df[new_col].cat.codes
        # Ensure that not known values are coded as OTHER
        df.loc[
            ~df[col].isin(self.vocabulary["A3A4"].keys()), new_col
        ] = self.vocabulary["A3A4"]["OTHER"]
        return df

    def _convert_B(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            [description]
        col : str
            column to translate

        Returns
        -------
        pd.DataFrame
            [description]
        """
        # Translate B1
        index_B = pd.IntervalIndex(list(self.vocabulary[col].keys()))
        new_col = "_" + col
        df[new_col] = pd.cut(df[col], index_B)
        df[new_col] = df[new_col].cat.codes
        df.loc[df[col] >= index_B.right.max(), new_col] = max(
            self.vocabulary[col].values()
        )
        df.loc[df[col] <= index_B.left.min(), new_col] = min(
            self.vocabulary[col].values()
        )

        return df

    def _convert_raw_data_to_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to translate data as extracted from spacy to the model codes.
        `A1` and `A2` are not translated cause are supposed to be already
        in good encoding.

        Parameters
        ----------
        df : pd.DataFrame
            It should have columns `['A3','A4','B1','B2']`

        Returns
        -------
        pd.DataFrame
        """
        df = self._convert_A(df, "A3")
        df = self._convert_A(df, "A4")
        df = self._convert_B(df, "B1")
        df = self._convert_B(df, "B2")
        return df

    def _convert_line_to_attribute(
        self, df: pd.DataFrame, expr: str, col: str
    ) -> pd.DataFrame:
        """
        Function to convert a line into an attribute (column) of the
        previous row. Particularly we use it to identify "\\n" and "\\n\\n"
        that are considered tokens, express this information as an attribute
        of the previous token.

        Parameters
        ----------
        df : pd.DataFrame
        expr : str
            pattern to search in the text. Ex.: "\\n"
        col : str
            name of the new column

        Returns
        -------
        pd.DataFrame
        """
        idx = df.TEXT.str.contains(expr)
        df.loc[idx, col] = True
        df[col] = df[col].fillna(False)
        df = self._shift_col(df, col, "TEMP_" + col, direction="backward")

        return df

    def _compute_a3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A3 (A4 respectively): typographic form  of left word (or right) :

        - All in capital letter
        - It starts with a capital letter
        - Starts by lowercase
        - It's a number
        - Strong punctuation
        - Soft punctuation
        - A number followed or preced by a punctuation (it's the case of enumerations)

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        df: pd.DataFrame with the columns `A3` and `A3_`

        """
        df = self._shift_col(
            df, "IS_PUNCT", "IS_PUNCT_+1", direction="backward", fill=False
        )
        df = self._shift_col(
            df, "IS_PUNCT", "IS_PUNCT_-1", direction="forward", fill=False
        )

        CONDITION1 = df.IS_UPPER
        CONDITION2 = df.SHAPE_.str.startswith("Xx", na=False)
        CONDITION3 = df.SHAPE_.str.startswith("x", na=False)
        CONDITION4 = df.IS_DIGIT
        STRONG_PUNCT = [".", ";", "..", "..."]
        CONDITION5 = (df.IS_PUNCT) & (df.TEXT.isin(STRONG_PUNCT))
        CONDITION6 = (df.IS_PUNCT) & (~df.TEXT.isin(STRONG_PUNCT))
        CONDITION7 = (df.IS_DIGIT) & (df["IS_PUNCT_+1"] | df["IS_PUNCT_-1"])  # discuss

        df["A3_"] = None
        df.loc[CONDITION1, "A3_"] = "UPPER"
        df.loc[CONDITION2, "A3_"] = "S_UPPER"
        df.loc[CONDITION3, "A3_"] = "LOWER"
        df.loc[CONDITION4, "A3_"] = "DIGIT"
        df.loc[CONDITION5, "A3_"] = "STRONG_PUNCT"
        df.loc[CONDITION6, "A3_"] = "SOFT_PUNCT"
        df.loc[CONDITION7, "A3_"] = "ENUMERATION"

        df = df.drop(columns=["IS_PUNCT_+1", "IS_PUNCT_-1"])
        df["A3_"] = df["A3_"].astype("category")

        df["A3_"] = df["A3_"].cat.add_categories("OTHER")
        df["A3_"].fillna("OTHER", inplace=True)

        df["A3"] = df["A3_"].cat.codes

        return df

    def _fit_M1(
        self,
        A1: pd.Series,
        A2: pd.Series,
        A3: pd.Series,
        A4: pd.Series,
        label: pd.Series,
    ):
        """Function to train M1 classifier (Naive Bayes)

        Parameters
        ----------
        A1 : pd.Series
            [description]
        A2 : pd.Series
            [description]
        A3 : pd.Series
            [description]
        A4 : pd.Series
            [description]
        label : pd.Series
            [description]

        """
        # Encode classes to OneHotEncoder representation
        encoder_A1_A2 = self._fit_encoder_2S(A1, A2)
        self.encoder_A1_A2 = encoder_A1_A2

        encoder_A3_A4 = self._fit_encoder_2S(A3, A4)
        self.encoder_A3_A4 = encoder_A3_A4

        # M1
        m1 = MultinomialNB(alpha=1)

        X = self._get_X_for_M1(A1, A2, A3, A4)
        m1.fit(X, label)
        self.m1 = m1

    def _fit_M2(self, B1: pd.Series, B2: pd.Series, label: pd.Series):
        """Function to train M2 classifier (Naive Bayes)

        Parameters
        ----------
        B1 : pd.Series
        B2 : pd.Series
        label : pd.Series
        """

        # Encode classes to OneHotEncoder representation
        encoder_B1 = self._fit_encoder_1S(B1)
        self.encoder_B1 = encoder_B1
        encoder_B2 = self._fit_encoder_1S(B2)
        self.encoder_B2 = encoder_B2

        # Multinomial Naive Bayes
        m2 = MultinomialNB(alpha=1)
        X = self._get_X_for_M2(B1, B2)
        m2.fit(X, label)
        self.m2 = m2

    def _get_X_for_M1(
        self, A1: pd.Series, A2: pd.Series, A3: pd.Series, A4: pd.Series
    ) -> np.ndarray:
        """Get X matrix for classifier

        Parameters
        ----------
        A1 : pd.Series
        A2 : pd.Series
        A3 : pd.Series
        A4 : pd.Series

        Returns
        -------
        np.ndarray
        """
        A1_enc = self._encode_series(self.encoder_A1_A2, A1)
        A2_enc = self._encode_series(self.encoder_A1_A2, A2)
        A3_enc = self._encode_series(self.encoder_A3_A4, A3)
        A4_enc = self._encode_series(self.encoder_A3_A4, A4)
        X = hstack([A1_enc, A2_enc, A3_enc, A4_enc])
        return X

    def _get_X_for_M2(self, B1: pd.Series, B2: pd.Series) -> np.ndarray:
        """Get X matrix for classifier

        Parameters
        ----------
        B1 : pd.Series
        B2 : pd.Series

        Returns
        -------
        np.ndarray
        """
        B1_enc = self._encode_series(self.encoder_B1, B1)
        B2_enc = self._encode_series(self.encoder_B2, B2)
        X = hstack([B1_enc, B2_enc])
        return X

    def _predict_M1(
        self, A1: pd.Series, A2: pd.Series, A3: pd.Series, A4: pd.Series
    ) -> Dict[str, Any]:
        """Use M1 for prediction

        Parameters
        ----------
        A1 : pd.Series
        A2 : pd.Series
        A3 : pd.Series
        A4 : pd.Series

        Returns
        -------
        Dict[str, Any]
        """
        X = self._get_X_for_M1(A1, A2, A3, A4)
        predictions = self.m1.predict(X)
        predictions_proba = self.m1.predict_proba(X)[:, 1]
        outputs = {"predictions": predictions, "predictions_proba": predictions_proba}
        return outputs

    def _predict_M2(self, B1: pd.Series, B2: pd.Series) -> Dict[str, Any]:
        """Use M2 for prediction

        Parameters
        ----------
        B1 : pd.Series
        B2 : pd.Series

        Returns
        -------
        Dict[str, Any]
        """
        X = self._get_X_for_M2(B1, B2)
        predictions = self.m2.predict(X)
        predictions_proba = self.m2.predict_proba(X)[:, 1]
        outputs = {"predictions": predictions, "predictions_proba": predictions_proba}
        return outputs

    def _fit_encoder_2S(self, S1: pd.Series, S2: pd.Series) -> OneHotEncoder:
        """Fit a one hot encoder with 2 Series. It concatenates the series and after it fits.

        Parameters
        ----------
        S1 : pd.Series
        S2 : pd.Series

        Returns
        -------
        OneHotEncoder
        """
        _S1 = _convert_series_to_array(S1)
        _S2 = _convert_series_to_array(S2)
        S = np.concatenate([_S1, _S2])
        encoder = self._fit_one_hot_encoder(S)
        return encoder

    def _fit_encoder_1S(self, S1: pd.Series) -> OneHotEncoder:
        """Fit a one hot encoder with 1 Series.

        Parameters
        ----------
        S1 : pd.Series

        Returns
        -------
        OneHotEncoder
        """
        _S1 = _convert_series_to_array(S1)
        encoder = self._fit_one_hot_encoder(_S1)
        return encoder

    def _encode_series(self, encoder: OneHotEncoder, S: pd.Series) -> np.ndarray:
        """Use the one hot encoder to transform a series.

        Parameters
        ----------
        encoder : OneHotEncoder
        S : pd.Series
            a series to encode (transform)

        Returns
        -------
        np.ndarray
        """
        _S = _convert_series_to_array(S)
        S_enc = encoder.transform(_S)
        return S_enc

    @classmethod
    def _retrieve_lines(cls, dfg: DataFrameGroupBy) -> DataFrameGroupBy:
        """Function to give a sentence_id to each token.

        Parameters
        ----------
        dfg : DataFrameGroupBy

        Returns
        -------
        DataFrameGroupBy
            Same DataFrameGroupBy with the column `SENTENCE_ID`
        """
        sentences_ids = np.arange(dfg.END_LINE.sum())
        dfg.loc[dfg.END_LINE, "SENTENCE_ID"] = sentences_ids
        dfg["SENTENCE_ID"] = dfg["SENTENCE_ID"].fillna(method="bfill")
        return dfg

    @classmethod
    def _create_vocabulary(cls, x: iterable) -> dict:
        """Function to create a vocabulary for attributes in the training set.

        Parameters
        ----------
        x : iterable

        Returns
        -------
        dict
        """
        v = {}

        for i, key in enumerate(x):
            v[key] = i

        return v

    @classmethod
    def _compute_B(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Function to compute B1 and B2

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        data = df.groupby(["DOC_ID", "SENTENCE_ID"]).agg(l=("LENGTH", "sum"))
        df_t = df.loc[df.END_LINE, ["DOC_ID", "SENTENCE_ID"]].merge(
            data, left_on=["DOC_ID", "SENTENCE_ID"], right_index=True, how="left"
        )

        stats_doc = df_t.groupby("DOC_ID").agg(mu=("l", "mean"), sigma=("l", "std"))
        stats_doc["sigma"].replace(
            0.0, 1.0, inplace=True
        )  # Replace the 0 std by unit std, otherwise it breaks the code.
        stats_doc["cv"] = stats_doc["sigma"] / stats_doc["mu"]

        df_t = df_t.drop(columns=["DOC_ID", "SENTENCE_ID"])
        df2 = df.merge(df_t, left_index=True, right_index=True, how="left")

        df2 = df2.merge(stats_doc, on=["DOC_ID"], how="left")
        df2["l_norm"] = (df2["l"] - df2["mu"]) / df2["sigma"]

        df2["cv_bin"] = pd.cut(df2["cv"], bins=10)
        df2["B2"] = df2["cv_bin"].cat.codes

        df2["l_norm_bin"] = pd.cut(df2["l_norm"], bins=10)
        df2["B1"] = df2["l_norm_bin"].cat.codes

        return df2

    @classmethod
    def _shift_col(
        cls, df: pd.DataFrame, col: str, new_col: str, direction="backward", fill=None
    ) -> pd.DataFrame:
        """Shifts a column one position into backward / forward direction.

        Parameters
        ----------
        df : pd.DataFrame
        col : str
            column to shift
        new_col : str
            column name to save the results
        direction : str, optional
            one of {"backward", "forward"}, by default "backward"
        fill : [type], optional
            , by default None

        Returns
        -------
        pd.DataFrame
            same df with `new_col` added.
        """
        df[new_col] = fill

        if direction == "backward":
            df.loc[df.index[:-1], new_col] = df[col].values[1:]

            different_doc_id = df["DOC_ID"].values[:-1] != df["DOC_ID"].values[1:]
            different_doc_id = np.append(different_doc_id, True)

        if direction == "forward":
            df.loc[df.index[1:], new_col] = df[col].values[:-1]
            different_doc_id = df["DOC_ID"].values[1:] != df["DOC_ID"].values[:-1]
            different_doc_id = np.append(True, different_doc_id)

        df.loc[different_doc_id, new_col] = fill
        return df

    @classmethod
    def _get_attributes(cls, doc: Doc, i=0):
        """Function to get the attributes of tokens of a spacy doc in a pd.DataFrame format.

        Parameters
        ----------
        doc : Doc
            spacy Doc
        i : int, optional
            document id, by default 0

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with one line per token. It has the following columns :
            `[
            "ORTH",
            "LOWER",
            "SHAPE",
            "IS_DIGIT",
            "IS_SPACE",
            "IS_UPPER",
            "IS_PUNCT",
            "LENGTH",
            ]`
        """
        attributes = [
            "ORTH",
            "LOWER",
            "SHAPE",
            "IS_DIGIT",
            "IS_SPACE",
            "IS_UPPER",
            "IS_PUNCT",
            "LENGTH",
        ]
        attributes_array = doc.to_array(attributes)
        attributes_df = pd.DataFrame(attributes_array, columns=attributes)
        attributes_df["DOC_ID"] = i
        boolean_attr = []
        for a in attributes:
            if a[:3] == "IS_":
                boolean_attr.append(a)
        attributes_df[boolean_attr] = attributes_df[boolean_attr].astype("boolean")
        return attributes_df

    @classmethod
    def _get_string(cls, _id: int, string_store: StringStore) -> str:
        """Returns the string corresponding to the token_id

        Parameters
        ----------
        _id : int
            token id
        string_store : StringStore
            spaCy Language String Store

        Returns
        -------
        str
            string representation of the token.
        """
        return string_store[_id]

    @classmethod
    def _fit_one_hot_encoder(cls, X: np.ndarray) -> OneHotEncoder:
        """Fit a one hot encoder.

        Parameters
        ----------
        X : np.ndarray
            of shape (n,1)

        Returns
        -------
        OneHotEncoder
        """
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(X)
        return encoder
