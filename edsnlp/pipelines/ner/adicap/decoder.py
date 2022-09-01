from edsnlp.utils.resources import get_adicap_dict


class AdicapDecoder:
    def __init__(
        self,
    ):
        self.decode_dict = get_adicap_dict()

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
