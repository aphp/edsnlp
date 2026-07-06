import importlib
import subprocess
import sys
import tarfile
import zipfile

import pytest

import edsnlp
from edsnlp.package import package


def test_blank_package(nlp, tmp_path):
    if not isinstance(nlp, edsnlp.Pipeline):
        pytest.skip("Only running for edsnlp.Pipeline")

    package(
        pipeline=nlp,
        root_dir=tmp_path,
        name="test-model-fail",
        metadata={},
    )

    nlp.package(
        root_dir=tmp_path,
        name="test-model",
        metadata={
            "description": "A test model",
            "authors": "Test Author <test.author@mail.com>",
        },
        distributions=["wheel"],
    )
    assert (tmp_path / "dist").is_dir()
    assert (tmp_path / "dist" / "test_model-0.1.0-py3-none-any.whl").is_file()
    assert not (tmp_path / "dist" / "test_model-0.1.0.tar.gz").is_file()


@pytest.mark.parametrize("package_name", ["my-test-model", None])
def test_package_with_files(nlp, tmp_path, package_name):
    if not isinstance(nlp, edsnlp.Pipeline):
        pytest.skip("Only running for edsnlp.Pipeline")

    nlp.to_disk(tmp_path / "model", exclude=set())

    ((tmp_path / "test_model").mkdir(parents=True))
    (tmp_path / "test_model" / "__init__.py").write_text('print("Hello World!")\n')
    (tmp_path / "test_model" / "empty_folder").mkdir()
    (tmp_path / "README.md").write_text(
        """\
<!-- INSERT -->
# Test Model
"""
    )
    (tmp_path / "pyproject.toml").write_text(
        """\
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "test-model"
version = "0.0.0"
description = "A test model"
authors = [
    {name = "Test Author", email = "test.author@mail.com"}
]
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "build"
]
"""
    )
    package(
        name=package_name,
        pipeline=tmp_path / "model",
        root_dir=tmp_path,
        check_dependencies=False,
        version="0.1.0",
        distributions=None,
        metadata={
            "description": "A new description",
            "authors": "Test Author <test.author@mail.com>",
        },
        readme_replacements={
            "<!-- INSERT -->": "Replaced !",
        },
    )

    module_name = "test_model" if package_name is None else "my_test_model"

    assert (tmp_path / "dist").is_dir()
    assert (tmp_path / "dist" / f"{module_name}-0.1.0.tar.gz").is_file()
    assert (tmp_path / "dist" / f"{module_name}-0.1.0-py3-none-any.whl").is_file()
    assert (tmp_path / "pyproject.toml").is_file()

    with zipfile.ZipFile(
        tmp_path / "dist" / f"{module_name}-0.1.0-py3-none-any.whl"
    ) as zf:
        # check files
        assert set(zf.namelist()) == {
            f"{module_name}-0.1.0.dist-info/METADATA",
            f"{module_name}-0.1.0.dist-info/RECORD",
            f"{module_name}-0.1.0.dist-info/WHEEL",
            f"{module_name}/__init__.py",
            f"{module_name}/artifacts/config.cfg",
            f"{module_name}/artifacts/meta.json",
            f"{module_name}/artifacts/tokenizer",
            "test_model/__init__.py",
        }
        # check description
        with zf.open(f"{module_name}-0.1.0.dist-info/METADATA") as f:
            assert b"A new description" in f.read()

    with tarfile.open(tmp_path / "dist" / f"{module_name}-0.1.0.tar.gz") as tf:
        # check files
        assert set(tf.getnames()) == {
            f"{module_name}-0.1.0/PKG-INFO",
            f"{module_name}-0.1.0/README.md",
            f"{module_name}-0.1.0/artifacts/config.cfg",
            f"{module_name}-0.1.0/artifacts/meta.json",
            f"{module_name}-0.1.0/artifacts/tokenizer",
            f"{module_name}-0.1.0/{module_name}/__init__.py",
            f"{module_name}-0.1.0/pyproject.toml",
            f"{module_name}-0.1.0/test_model/__init__.py",
        }
        # check description
        with tf.extractfile(f"{module_name}-0.1.0/PKG-INFO") as f:
            assert b"A new description" in f.read()

        with tf.extractfile(f"{module_name}-0.1.0/README.md") as f:
            assert b"Replaced !" in f.read()

    # pip install the whl file
    (tmp_path / "site-packages").mkdir(exist_ok=True)
    subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-vvv",
            "--target",
            str(tmp_path / "site-packages"),
            str(tmp_path / "dist" / f"{module_name}-0.1.0-py3-none-any.whl"),
        ],
        stderr=subprocess.STDOUT,
    )

    site_packages = tmp_path / "site-packages"
    sys.path.insert(0, str(site_packages))
    # check site-package files
    files = {str(f.relative_to(site_packages)) for f in set(site_packages.rglob("*"))}
    assert files >= {
        f"{module_name}/artifacts",
        f"{module_name}/artifacts/config.cfg",
        f"{module_name}/artifacts/meta.json",
        f"{module_name}/artifacts/tokenizer",
    }

    module = importlib.import_module(module_name)

    with open(module.__file__) as f:
        assert f.read() == (
            ('print("Hello World!")\n' if package_name is None else "")
            + """
# -----------------------------------------
# This section was autogenerated by edsnlp
# -----------------------------------------

import edsnlp
from pathlib import Path
from typing import Optional, Dict, Any

__version__ = '0.1.0'

def load(
    overrides: Optional[Dict[str, Any]] = None,
) -> edsnlp.Pipeline:
    path_outside = Path(__file__).parent / "../artifacts"
    path_inside = Path(__file__).parent / "artifacts"
    path = path_inside if path_inside.exists() else path_outside
    model = edsnlp.load(path, overrides=overrides)
    return model
"""
        )
    module.load()
    edsnlp.load(module_name)


def test_package_dependency_mode_with_code_wheel_and_local_index(tmp_path):
    (tmp_path / "project_code").mkdir()
    (tmp_path / "project_code" / "__init__.py").write_text("")
    (tmp_path / "project_code" / "pipes.py").write_text(
        """\
from edsnlp.core.registries import registry


@registry.factories.register("project_code.dummy")
def create_component(nlp, name):
    def pipe(doc):
        return doc

    return pipe
"""
    )
    (tmp_path / "pyproject.toml").write_text(
        f"""\
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "project-code"
version = "1.4.0"
description = "A test code package"
authors = [
    {{name = "Test Author", email = "test.author@mail.com"}}
]
requires-python = ">=3.10"

[project.entry-points."edsnlp_factories"]
"project_code.dummy" = "project_code.pipes:create_component"

[[tool.uv.index]]
name = "local"
url = "{(tmp_path / "simple").as_uri()}"
publish-url = "{(tmp_path / "upload").as_uri()}"
explicit = true

[tool.setuptools.packages.find]
where = ["."]
include = ["project_code*"]
"""
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path / "dist"),
        ],
        cwd=tmp_path,
    )
    code_wheel = next((tmp_path / "dist").glob("project_code-1.4.0-*.whl"))
    simple_package = tmp_path / "simple" / "project-code"
    simple_package.mkdir(parents=True)
    (simple_package / "index.html").write_text(
        "\n".join(
            [
                '<a href="not-a-wheel.txt">not-a-wheel.txt</a>',
                '<a href="other_package-1.4.0-py3-none-any.whl">'
                "other_package-1.4.0-py3-none-any.whl</a>",
                f'<a href="../../dist/{code_wheel.name}">{code_wheel.name}</a>',
            ]
        )
    )

    sys.path.insert(0, str(tmp_path))
    try:
        import project_code.pipes  # noqa: F401

        nlp = edsnlp.blank("eds")
        nlp.add_pipe("project_code.dummy")
        nlp.to_disk(tmp_path / "model", exclude=set())
    finally:
        sys.path.remove(str(tmp_path))

    package(
        name="project-code-model",
        pipeline=tmp_path / "model",
        root_dir=tmp_path,
        version="2026.7.4",
        distributions=["wheel"],
        code="dependency",
        code_check="error",
    )

    model_wheel = tmp_path / "dist" / "project_code_model-2026.7.4-py3-none-any.whl"
    assert model_wheel.is_file()
    with zipfile.ZipFile(model_wheel) as zf:
        names = set(zf.namelist())
        assert "project_code/__init__.py" not in names
        assert "project_code/pipes.py" not in names
        assert "project_code_model/artifacts/config.cfg" in names
        metadata = zf.read("project_code_model-2026.7.4.dist-info/METADATA").decode()
        assert "Requires-Dist: project-code<1.5,>=1.4" in metadata
        meta = zf.read("project_code_model/artifacts/meta.json").decode()
        assert '"dependency": "project-code>=1.4,<1.5"' in meta

    with pytest.raises(RuntimeError, match="Could not find uv index"):
        package(
            name="project-code-model-missing-index",
            pipeline=tmp_path / "model",
            root_dir=tmp_path,
            version="2026.7.5",
            distributions=["wheel"],
            code="dependency",
            code_check="error",
            publish_index="missing",
        )

    with pytest.raises(RuntimeError, match="Code dependency"):
        package(
            name="project-code-model-missing-dep",
            pipeline=tmp_path / "model",
            root_dir=tmp_path,
            version="2026.7.5",
            distributions=["wheel"],
            code="dependency",
            code_dependency="project-code>=2,<3",
            code_check="error",
            publish_index="local",
        )

    (tmp_path / "project_code" / "pipes.py").write_text(
        (tmp_path / "project_code" / "pipes.py").read_text() + "\nVALUE = 1\n"
    )
    (tmp_path / "project_code" / "extra.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="Local project code differs"):
        package(
            name="project-code-model-drift",
            pipeline=tmp_path / "model",
            root_dir=tmp_path,
            version="2026.7.5",
            distributions=["wheel"],
            code="dependency",
            code_check="error",
            publish_index="local",
        )


def test_package_none_mode_excludes_project_code(nlp, tmp_path):
    if not isinstance(nlp, edsnlp.Pipeline):
        pytest.skip("Only running for edsnlp.Pipeline")

    nlp.to_disk(tmp_path / "model", exclude=set())
    (tmp_path / "project_code").mkdir()
    (tmp_path / "project_code" / "__init__.py").write_text('VALUE = "unused"\n')
    (tmp_path / "pyproject.toml").write_text(
        """\
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "project-code"
version = "1.4.0"
description = "A test code package"
authors = [
    {name = "Test Author", email = "test.author@mail.com"}
]
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["project_code*"]
"""
    )

    with pytest.warns(UserWarning, match="code='none'"):
        package(
            name="artifact-only-model",
            pipeline=tmp_path / "model",
            root_dir=tmp_path,
            version="2026.7.4",
            distributions=["wheel"],
            code="none",
        )

    model_wheel = tmp_path / "dist" / "artifact_only_model-2026.7.4-py3-none-any.whl"
    with zipfile.ZipFile(model_wheel) as zf:
        names = set(zf.namelist())
        assert "project_code/__init__.py" not in names
        assert "artifact_only_model/artifacts/config.cfg" in names
