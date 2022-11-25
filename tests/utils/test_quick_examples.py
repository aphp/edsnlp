from edsnlp.viz.quick_examples import QuickExample


def test_quick_example_text(nlp, capsys):
    text = "Le patient présente une anomalie"

    E = QuickExample(nlp, extensions=["sent"])
    E(text)

    captured = capsys.readouterr()

    assert "Entity" in captured.out
    assert "Source" in captured.out
    assert "patient" in captured.out
    assert "anomalie" in captured.out
    assert "sent" in captured.out


def test_quick_example_doc(nlp, capsys):
    text = "Le patient présente une anomalie"

    E = QuickExample(nlp, extensions=["sent"])
    E(nlp(text))

    captured = capsys.readouterr()

    assert "Entity" in captured.out
    assert "Source" in captured.out
    assert "patient" in captured.out
    assert "anomalie" in captured.out
    assert "sent" in captured.out


def test_quick_example_df(nlp):
    text = "Le patient présente une anomalie"

    E = QuickExample(nlp, extensions=["sent"])
    df = E(text, as_dataframe=True)
    assert df.shape == (2, 8)
