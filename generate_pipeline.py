import typer
import shutil
import stringcase

from pathlib import Path


def main(name: str):
    print(f"{name}")
    dir_path = Path(f"./edsnlp/pipelines/ner/comorbidities/{name}")
    dir_path.mkdir()

    shutil.copyfile(
        "./edsnlp/pipelines/ner/comorbidities/diabetes/diabetes.py",
        dir_path / f"{name}.py",
    )

    shutil.copyfile(
        "./edsnlp/pipelines/ner/comorbidities/diabetes/factory.py",
        dir_path / f"factory.py",
    )

    shutil.copyfile(
        "./edsnlp/pipelines/ner/comorbidities/diabetes/patterns.py",
        dir_path / f"patterns.py",
    )

    for file in ["factory", name]:
        p = dir_path / f"{file}.py"
        content = p.open("r").read()
        content = content.replace("Diabetes", stringcase.pascalcase(name)).replace(
            "diabetes", name
        )
        p.open("w").write(content)

    p = Path("./edsnlp/pipelines/factories.py")
    content = p.open("r").read()
    content = (
        content
        + f"from .ner.comorbidities.{name}.factory import create_component as {name}\n"
    )
    p.open("w").write(content)

    p = Path("./setup.py")
    content = p.open("r").read()
    content = content.replace(
        '"diabetes = edsnlp.components:diabetes"',
        (
            '"diabetes = edsnlp.components:diabetes",'
            + "\n    "
            + f'"{name} = edsnlp.components:{name}"'
        ),
    )
    p.open("w").write(content)


if __name__ == "__main__":
    typer.run(main)
