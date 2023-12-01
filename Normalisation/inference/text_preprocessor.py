from unidecode import unidecode
import re

class TextPreprocessor():
    def __init__(
        self,
        cased,
        stopwords
    ):
        self.cased = cased
        self.regex_stopwords = re.compile(r"\b(?:" + "|".join(stopwords) + r")\b", re.IGNORECASE)
        self.regex_special_characters = re.compile(r"[^a-zA-Z0-9\s]", re.IGNORECASE)
    
    def normalize(self, txt, remove_stopwords, remove_special_characters):
        if not self.cased:
            txt = unidecode(
                txt.lower()
                .replace("-", " ")
                .replace("ag ", "antigene ")
                .replace("ac ", "anticorps ")
                .replace("antigenes ", "antigene ")
            )
        else:
            txt = unidecode(
                txt.replace("-", " ")
                .replace("ag ", "antigene ")
                .replace("ac ", "anticorps ")
                .replace("antigenes ", "antigene ")
                .replace("Ag ", "Antigene ")
                .replace("Ac ", "Anticorps ")
                .replace("Antigenes ", "Antigene ")
            )
        if remove_stopwords:
            txt = self.regex_stopwords.sub("", txt)
        if remove_special_characters:
            txt = self.regex_special_characters.sub(" ", txt)
        return re.sub(" +", " ", txt).strip()
    
    def __call__(self, text, remove_stopwords = False, remove_special_characters = False):
        return self.normalize(text, remove_stopwords, remove_special_characters)
