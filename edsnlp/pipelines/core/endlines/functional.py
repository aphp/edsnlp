import os

import numpy as np
import pandas as pd


def get_dir_path(file):
    path_file = os.path.dirname(os.path.realpath(file))
    return path_file


def build_path(file, relative_path):
    """
    Function to build an absolut path.

    Parameters
    ----------
    file: main file from where we are calling. It could be __file__
    relative_path: str,
        relative path from the main file to the desired output

    Returns
    -------
    path: absolute path
    """
    dir_path = get_dir_path(file)
    path = os.path.abspath(os.path.join(dir_path, relative_path))
    return path


def _convert_series_to_array(s: pd.Series) -> np.ndarray:
    """Converts pandas series of n elements to an array of shape (n,1).

    Parameters
    ----------
    s : pd.Series

    Returns
    -------
    np.ndarray
    """
    X = s.to_numpy().reshape(-1, 1).astype("O")  # .astype(np.int64)
    return X
