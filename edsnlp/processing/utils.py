import types

from edsnlp.data.converters import set_current_tokenizer


def apply_basic_pipes(docs, pipes):
    for name, pipe, kwargs, tok in pipes:
        with set_current_tokenizer(tok):
            if hasattr(pipe, "batch_process"):
                docs = pipe.batch_process(docs)
            else:
                results = []
                for doc in docs:
                    res = pipe(doc, **kwargs)
                    if isinstance(res, types.GeneratorType):
                        results.extend(res)
                    else:
                        results.append(res)
                docs = results
    return docs
