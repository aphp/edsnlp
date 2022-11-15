import regex
import spacy
from loguru import logger
from spacy import Vocab
from spacy.lang.fr import French, FrenchDefaults
from spacy.lang.fr.lex_attrs import LEX_ATTRS
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.lang.fr.syntax_iterators import SYNTAX_ITERATORS
from spacy.tokens import Doc
from spacy.util import DummyTokenizer


class EDSDefaults(FrenchDefaults):
    """
    Defaults for the EDSLanguage class
    Mostly identical to the FrenchDefaults, but
    without tokenization info
    """

    tokenizer_exceptions = {}
    infixes = []
    lex_attr_getters = LEX_ATTRS
    syntax_iterators = SYNTAX_ITERATORS
    stop_words = STOP_WORDS
    config = FrenchDefaults.config.merge(
        {
            "nlp": {"tokenizer": {"@tokenizers": "eds.tokenizer"}},
        }
    )


@spacy.registry.languages("eds")
class EDSLanguage(French):
    """
    French clinical language.
    It is shipped with the `EDSTokenizer` tokenizer that better handles
    tokenization for French clinical documents
    """

    lang = "eds"
    Defaults = EDSDefaults
    default_config = Defaults


class EDSTokenizer(DummyTokenizer):
    def __init__(self, vocab: Vocab) -> None:
        """
        Tokenizer class for French clinical documents.
        It better handles tokenization around:
        - numbers: "ACR5" -> ["ACR", "5"] instead of ["ACR5"]
        - newlines: "\n \n \n" -> ["\n", "\n", "\n"] instead of ["\n \n \n"]
        and should be around 5-6 times faster than its standard French counterpart.

        Parameters
        ----------
        vocab: Vocab
            The spacy vocabulary
        """
        self.vocab = vocab
        punct = "[:punct:]" + "\"'ˊ＂〃ײ᳓″״‶˶ʺ“”˝"
        num_like = r"\d+(?:[.,]\d+)?"
        default = rf"[^\d{punct}'\n[[:space:]]+(?:['ˊ](?=[[:alpha:]]|$))?"
        self.word_regex = regex.compile(
            rf"({num_like}|[{punct}]|[\n\r\t]|[^\S\r\n\t]+|{default})([^\S\r\n\t])?"
        )

    def __call__(self, text: str) -> Doc:
        """
        Tokenizes the text using the EDSTokenizer

        Parameters
        ----------
        text: str

        Returns
        -------
        Doc

        """
        last = 0
        words = []
        whitespaces = []
        for match in self.word_regex.finditer(text):
            begin, end = match.start(), match.end()
            if last != begin:
                logger.warning(
                    "Missed some characters during"
                    + f" tokenization between {last} and {begin}: "
                    + text[last - 10 : last]
                    + "|"
                    + text[last:begin]
                    + "|"
                    + text[begin : begin + 10],
                )
            last = end
            words.append(match.group(1))
            whitespaces.append(bool(match.group(2)))
        return Doc(self.vocab, words=words, spaces=whitespaces)


@spacy.registry.tokenizers("eds.tokenizer")
def create_eds_tokenizer():
    """
    Creates a factory that returns new EDSTokenizer instances

    Returns
    -------
    EDSTokenizer
    """

    def eds_tokenizer_factory(nlp):
        return EDSTokenizer(nlp.vocab)

    return eds_tokenizer_factory


__all__ = ["EDSLanguage"]
