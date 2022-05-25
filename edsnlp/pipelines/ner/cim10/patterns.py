from typing import Dict, List

import pandas as pd

from edsnlp import BASE_DIR


def get_patterns() -> Dict[str, List[str]]:
    df = pd.read_csv(BASE_DIR / "resources" / "cim10.csv.gz")
    df = pd.concat(
        [
            df[["code", "short"]].rename(columns={"short": "patterns"}),
            df[["code", "long"]].rename(columns={"long": "patterns"}),
        ]
    )

    patterns = df.groupby("code")["patterns"].agg(list).to_dict()

    return patterns
