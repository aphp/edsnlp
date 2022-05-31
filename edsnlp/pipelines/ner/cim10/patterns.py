from typing import Dict, List

import pandas as pd

from edsnlp import BASE_DIR


def get_patterns() -> Dict[str, List[str]]:
    df = pd.read_csv(BASE_DIR / "resources" / "cim10.csv.gz")

    df["code_pattern"] = df["code"]
    df["code_point"] = df["code"].str[:2] + "." + df["code"].str[2:]
    df["code_space"] = df["code"].str[0] + " " + df["code"].str[1:]
    df["code_space_point"] = (
        df["code"].str[0] + " " + df["code"].str[1] + "." + df["code"].str[2:]
    )

    df = pd.concat(
        [
            df[["code", "short"]].rename(columns={"short": "patterns"}),
            df[["code", "long"]].rename(columns={"long": "patterns"}),
            df[["code", "code_pattern"]].rename(columns={"code_pattern": "patterns"}),
            df[["code", "code_point"]].rename(columns={"code_point": "patterns"}),
            df[["code", "code_space"]].rename(columns={"code_space": "patterns"}),
            df[["code", "code_space_point"]].rename(
                columns={"code_space_point": "patterns"}
            ),
        ]
    )

    patterns = df.groupby("code")["patterns"].agg(list).to_dict()

    return patterns
