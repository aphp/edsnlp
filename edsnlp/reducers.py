import contextlib
import copyreg
from weakref import WeakSet

from dill._dill import _is_builtin_module
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab

_pickle_memo = WeakSet()


def _reduce_doc(doc):
    seen = doc in _pickle_memo

    if not seen:
        # uid = uuid.uuid4()
        _pickle_memo.add(doc)
        # -> [uid, need to encode body]
        return (_add_back_user_data_and_hooks, (doc, doc.user_data, doc.user_hooks))

    # uid, need_to_encode_body = _pickle_memo[doc]
    # if not need_to_encode_body:
    #     return (_rebuild_doc, (uid,))

    array_head = Doc._get_array_attrs()
    strings = set()
    for token in doc:
        strings.add(token.tag_)
        strings.add(token.lemma_)
        strings.add(str(token.morph))
        strings.add(token.dep_)
        strings.add(token.ent_type_)
        strings.add(token.ent_kb_id_)
        strings.add(token.ent_id_)
        strings.add(token.norm_)
    for group in doc.spans.values():
        for span in group:
            strings.add(span.label_)
            if span.kb_id in span.doc.vocab.strings:
                strings.add(span.kb_id_)
            if span.id in span.doc.vocab.strings:
                strings.add(span.id_)
    data = {
        "vocab": doc.vocab,
        "text": doc.text,
        "array_head": array_head,
        "array_body": doc.to_array(array_head),
        "sentiment": doc.sentiment,
        "tensor": doc.tensor,
        "cats": doc.cats,
        "spans": doc.spans.to_bytes(),
        "strings": list(strings),
        "has_unknown_spaces": doc.has_unknown_spaces,
    }
    _pickle_memo.remove(doc)
    return (_rebuild_doc, (data,))


def _rebuild_doc(data):
    doc = Doc(Vocab())
    doc.from_dict(data)
    return doc


def _add_back_user_data_and_hooks(doc, user_data, user_hooks):
    doc.user_data = user_data
    doc.user_hooks = user_hooks
    return doc


def _reduce_span(span):
    data = {
        "start": span.start,
        "end": span.end,
        "label": span.label_,
        "kb_id": span.kb_id_,
        "id": span.id_,
        "doc": span.doc,
    }
    return (_rebuild_span, (data,))


def _rebuild_span(data):
    return Span(
        doc=data["doc"],
        start=data["start"],
        end=data["end"],
        label=data["label"],
        kb_id=data["kb_id"],
        span_id=data["id"],
    )


def _reduce_token(token):
    return (
        _rebuild_token,
        (
            token.doc,
            token.i,
        ),
    )


def _rebuild_token(doc, i):
    """Import the token contents from a dictionary.

    doc (Doc): The parent document.
    i (int): The index of the token in the document.
    RETURNS (Token): The deserialized `Token`.
    """
    return doc[i]


def custom_is_builtin_module(module):
    if not hasattr(module, "__file__"):
        return True
    if module.__file__ is None:
        return False
    # Hack to avoid copying the full module dict
    return True


@contextlib.contextmanager
def pickler_dont_save_module_dict():
    old_is_builtin_module_code = _is_builtin_module.__code__
    old_is_builtin_module_defaults = _is_builtin_module.__defaults__

    _is_builtin_module.__code__ = custom_is_builtin_module.__code__
    _is_builtin_module.__defaults__ = custom_is_builtin_module.__defaults__

    yield

    _is_builtin_module.__code__ = old_is_builtin_module_code
    _is_builtin_module.__defaults__ = old_is_builtin_module_defaults


copyreg.pickle(Doc, _reduce_doc)
copyreg.pickle(Span, _reduce_span)
copyreg.pickle(Token, _reduce_token)
