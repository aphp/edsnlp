import pytest

pytest.importorskip("umls_downloader", reason="umls_downloader package not found")

import os  # noqa: E402
from pathlib import Path  # noqa: E402

from spacy.language import Language  # noqa: E402

from edsnlp.pipelines.ner.umls.patterns import get_path, get_patterns  # noqa: E402
from edsnlp.utils.examples import parse_example  # noqa: E402

examples = [
    "Grosse <ent umls=C0010200>toux</ent> : Le malade a été mordu "
    "<ent umls=C1257901>par</ent> des <ent umls=C0002668>Amphibiens</ent>"
    " sous le <ent umls=C0022742>genou</ent>"
]
pattern_config = {"lang": ["FRE"], "sources": ["MSHFRE"]}


@pytest.mark.skipif(not os.getenv("UMLS_API_KEY"), reason="No UMLS_API_KEY given")
def test_get_patterns():

    path, _, _ = get_path(pattern_config)
    assert isinstance(path, Path)

    api_key = os.getenv("UMLS_API_KEY")

    try:
        patterns = get_patterns(pattern_config)
    except AttributeError:
        if not api_key:
            raise Exception(
                "Missing UMLS_API_KEY env variable to download the UMLS. "
                f"UMLS cannot be found at path: {path}.\nTo skip the test, "
                f"uninstall the umls_downloader package."
            )
        raise

    assert path.exists()

    assert len(patterns) == 48587


@pytest.mark.skipif(not os.getenv("UMLS_API_KEY"), reason="No UMLS_API_KEY given")
def test_add_pipe(blank_nlp: Language):
    path, _, _ = get_path(pattern_config)
    if not path.exists():
        pytest.xfail(
            "The umls_downloader package is installed but "
            f"UMLS cannot be found at path: {path}"
        )

    blank_nlp.add_pipe("eds.umls", config=dict(pattern_config=pattern_config))

    assert "eds.umls" in blank_nlp.pipe_names

    for text, entities in map(parse_example, examples):
        doc = blank_nlp(text)
        assert len(doc.ents) == len(entities)

        for ent, entity in zip(doc.ents, entities):
            assert ent.text == text[entity.start_char : entity.end_char]
            assert ent._.umls == ent.kb_id_ == entity.modifiers[0].value
