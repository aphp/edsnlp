import importlib
import subprocess
import sys
import tarfile
import zipfile

import pytest

import edsnlp
from edsnlp.package import package


def test_blank_package(nlp, tmp_path):
    # Missing metadata makes poetry fail due to missing author / description
    if not isinstance(nlp, edsnlp.Pipeline):
        pytest.skip("Only running for edsnlp.Pipeline")

    package(
        pipeline=nlp,
        root_dir=tmp_path,
        name="test-model-fail",
        metadata={},
        project_type="poetry",
    )

    nlp.package(
        root_dir=tmp_path,
        name="test-model",
        metadata={
            "description": "A test model",
            "authors": "Test Author <test.author@mail.com>",
        },
        project_type="poetry",
        distributions=["wheel"],
    )
    assert (tmp_path / "dist").is_dir()
    assert (tmp_path / "dist" / "test_model-0.1.0-py3-none-any.whl").is_file()
    assert not (tmp_path / "dist" / "test_model-0.1.0.tar.gz").is_file()


@pytest.mark.parametrize("package_name", ["my-test-model", None])
@pytest.mark.parametrize("manager", ["poetry", "setuptools"])
def test_package_with_files(nlp, tmp_path, package_name, manager):
    if not isinstance(nlp, edsnlp.Pipeline):
        pytest.skip("Only running for edsnlp.Pipeline")

    nlp.to_disk(tmp_path / "model", exclude=set())

    ((tmp_path / "test_model").mkdir(parents=True))
    (tmp_path / "test_model" / "__init__.py").write_text('print("Hello World!")\n')
    (tmp_path / "README.md").write_text(
        """\
<!-- INSERT -->
# Test Model
"""
    )
    if manager == "poetry":
        (tmp_path / "pyproject.toml").write_text(
            """\
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "test-model"
version = "0.0.0"
description = "A test model"
authors = ["Test Author <test.author@mail.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.7"
build = "*"  # sample light package to install
"""
        )
    elif manager == "setuptools":
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
requires-python = ">=3.7"

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
