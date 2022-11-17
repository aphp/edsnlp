import os
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pystow
from tqdm import tqdm
from umls_downloader import download_umls

PATTERN_VERSION = "0.7.0"
UMLS_VERSION = "2022AA"


def get_patterns(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Load the UMLS terminology patterns.

    Parameters
    ----------
    config : dict[list]
        Languages and sources to select from the whole terminology.
        For both keys, None will select all values.

    Return
    ------
    patterns : dict[list]
        The mapping between CUI codes and their synonyms.

    Notes
    -----
    When run for the first time, this method will download the
    entire UMLS file and store it at ~/.data/bio/umls/2022AA/.
    Therefore the second run will be significantly faster than
    the first one.
    """

    path, module, filename = get_path(config)

    if path.exists():
        print(f"Loading {filename} from {module.base}")
        return module.load_pickle(name=filename)
    else:
        patterns = download_and_agg_umls(config)
        module.dump_pickle(name=filename, obj=patterns)
        print(f"Saved patterns into {module.base / filename}")
        return patterns


def get_path(config: Dict[str, Any]) -> Tuple[Path, pystow.impl.Module, str]:
    """Get the path, module and filename of the UMLS file.

    Parameters
    ----------
    config : dict[list]
        Languages and sources to select from the whole terminology.
        For both keys, None will select all values.

    Return
    ------
    path, module, filename : pathlib.Path, pystow.module, str

    Notes
    -----
    `get_path` will convert the config dict into a pretty filename.

    Examples
    --------
    >>> config = {"languages": ["FRE", "ENG"], "sources": None}
    >>> print(get_path(config))
    .data/bio/umls/2022AA/languagesFRE-ENG_sourcesNone.pkl"
    """
    config_txt = ""
    for k, v in config.items():
        if isinstance(v, (list, tuple)):
            v = "-".join(v)
        _config_txt = f"{k}{v}"
        if config_txt:
            _config_txt = f"_{_config_txt}"
        config_txt += _config_txt

    filename = f"{PATTERN_VERSION}_{config_txt}.pkl"
    module = pystow.module("bio", "umls", UMLS_VERSION)
    path = module.base / filename

    return path, module, filename


def download_and_agg_umls(config) -> Dict[str, List[str]]:
    """Download the UMLS if not exist and create a mapping
    between CUI code and synonyms.

    Parameters
    ----------
    config : dict[list]
        Languages and sources to select from the whole terminology.
        For both keys, None will select all values.

    Return
    ------
    patterns : dict[list]
        The mapping between CUI codes and their synonyms.

    Notes
    -----
    Performs filtering on the returned mapping only, not the downloaded
    resource.
    """

    api_key = os.getenv("UMLS_API_KEY")
    if not api_key:
        warnings.warn(
            "You need to define UMLS_API_KEY to download the UMLS. "
            "Get a key by creating an account at "
            "https://uts.nlm.nih.gov/uts/signup-login"
        )

    path = download_umls(version=UMLS_VERSION, api_key=api_key)

    # https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/ # noqa
    header = [
        "CUI",
        "LAT",
        "TS",
        "LUI",
        "STT",
        "SUI",
        "ISPREF",
        "AUI",
        "SAUI",
        "SCUI",
        "SDUI",
        "SAB",
        "TTY",
        "CODE",
        "STR",
        "STRL",
        "SUPPRESS",
    ]
    header_to_idx = dict(zip(header, range(len(header))))
    patterns = defaultdict(list)

    languages = config.get("languages")
    sources = config.get("sources")

    with zipfile.ZipFile(path) as zip_file:
        with zip_file.open("MRCONSO.RRF", mode="r") as file:
            for row in tqdm(
                file.readlines(), desc="Loading 'MRCONSO.RRF' into a dictionnary"
            ):
                row = row.decode("utf-8").split("|")

                if (languages is None or row[header_to_idx["LAT"]] in languages) and (
                    sources is None or row[header_to_idx["SAB"]] in sources
                ):
                    cui = row[header_to_idx["CUI"]]
                    synonym = row[header_to_idx["STR"]]
                    patterns[cui].append(synonym)

    return patterns
