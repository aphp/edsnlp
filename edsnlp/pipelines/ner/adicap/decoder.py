from typing import List

from edsnlp.utils.resources import get_adicap_data


class AdicapDecoder:
    def __init__(
        self,
        dict_keys: List[str] = ["D1", "D2", "D3", "D4", "D5"],
    ):
        self.dict_keys = dict_keys
        self.df = get_adicap_data()

        self.get_decode_dict()

    def parse_each_dict(self, dictionaryCode: str):
        d_spec = self.df.query(f"dictionaryCode=='{dictionaryCode}'")

        decode_d_spec = {}

        for code, label in d_spec[["code", "label"]].values:
            decode_d_spec[code] = label

        d_value = decode_d_spec.pop(dictionaryCode)

        return dict(label=d_value, codes=decode_d_spec)

    def get_decode_dict(self):
        decode_dict = {}
        for key in self.dict_keys:

            decode_dict[key] = self.parse_each_dict(dictionaryCode=key)

        self.decode_dict = decode_dict

    def decode_adicap(self, code: str):
        exploded = list(code)
        decoded = {
            "code": code,
            "sampling_mode": self.decode_dict["D1"]["codes"].get(exploded[0]),
            "technic": self.decode_dict["D2"]["codes"].get(exploded[1]),
            "organ": self.decode_dict["D3"]["codes"].get("".join(exploded[2:4])),
            "non_tumoral_pathology": self.decode_dict["D4"]["codes"].get(
                "".join(exploded[4:8])
            ),
            "tumoral_pathology": self.decode_dict["D5"]["codes"].get(
                "".join(exploded[4:8])
            ),
            "behaviour_type": self.decode_dict["D5"]["codes"].get(exploded[5]),
        }
        return decoded
