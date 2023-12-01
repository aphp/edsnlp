# This script exists because spaCy's package script cannot natively
# - export a package (for medic rules) as well as the model weights
# - keep the same requirements as those listed in the pyproject.toml file
# - include package_data for the medic rules, listed also in the pyproject
# Therefore, we in this script, we build the model package using most of spaCy's code
# and build a new pyproject.toml derived from the main one (at the root of the repo)
# and use poetry to build the model instead of `python setup.py`.

import os.path
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import srsly
import toml
from spacy import util
from spacy.cli._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, string_to_list
from spacy.cli.package import (
    FILENAMES_DOCS,
    TEMPLATE_MANIFEST,
    _is_permitted_package_name,
    create_file,
    generate_meta,
    generate_readme,
    get_build_formats,
    get_meta,
    has_wheel,
)
from spacy.schemas import ModelMetaSchema, validate
from typer import Typer
from wasabi import Printer

app = Typer()


TEMPLATE_INIT = """
from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta

{imports}

__version__ = {version}


def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)
""".lstrip()


def make_pyproject_toml(
    pyproject_path: str,
    name: str,
    package_name: str,
    package_data: str,
) -> str:
    """
    Creates a new pyproject.toml config for the generated model
    from the root poetry-based pyproject.toml file
    For that we:
    - adapt paths to the new structure (nested under the new model name)
    - change the pyproject name
    - include any original included files as well as the model weights

    Parameters
    ----------
    pyproject_path: str
        Path to the root level pyproject.toml path
    name: str
        Model name
    package_name: str
        Package directory name
    package_data: str
        Package data name in the above directory

    Returns
    -------
    str
    The string content of the new pyproject.toml file
    """
    package_name = Path(package_name)
    pyproject_text = Path(pyproject_path).read_text()
    pyproject_data = toml.loads(pyproject_text)
    print(pyproject_data)
    pyproject_data["tool"]["poetry"]["name"] = name
    new_includes = [
        str(package_name / include)
        for include in pyproject_data["tool"]["poetry"]["include"]
    ] + [
        str(package_name / "**/*.py"),
        str(package_name / package_data / "**/*"),
        str("**/meta.json"),
    ]
    pyproject_data["tool"]["poetry"]["include"] = new_includes
    for key, plugins in pyproject_data["tool"]["poetry"]["plugins"].items():
        new_plugins = {}
        for value, path in plugins.items():
            new_plugins[value] = f"{package_name}.{path}"
        plugins.clear()
        plugins.update(new_plugins)
    return toml.dumps(pyproject_data)


# fmt: off
@app.command()
def package_medic_cli(
    input_dir: Path = Arg(..., help="Directory with pipeline data", exists=True, file_okay=False),  # noqa: E501
    output_dir: Path = Arg(..., help="Output parent directory", exists=True, file_okay=False),  # noqa: E501
    code_paths: str = Opt("", "--code", "-c", help="Comma-separated paths to Python file with additional code (registered functions) to be included in the package"),  # noqa: E501
    meta_path: Optional[Path] = Opt(None, "--meta-path", "--meta", "-m", help="Path to meta.json", exists=True, dir_okay=False),  # noqa: E501
    create_meta: bool = Opt(False, "--create-meta", "-C", help="Create meta.json, even if one exists"),  # noqa: E501
    name: Optional[str] = Opt(None, "--name", "-n", help="Package name to override meta"),  # noqa: E501
    version: Optional[str] = Opt(None, "--version", "-v", help="Package version to override meta"),  # noqa: E501
    build: str = Opt("sdist", "--build", "-b", help="Comma-separated formats to build: sdist and/or wheel, or none."),  # noqa: E501
    force: bool = Opt(False, "--force", "-f", "-F", help="Force overwriting existing data in output directory"),  # noqa: E501
):
    # fmt: on
    """
    Adapted from spaCy package CLI command (documentation copied below)
    This script exists spaCy's standard script cannot natively
    - export a package (for medic rules) as well as the model weights
    - keep the same requirements as those listed in the pyproject.toml file
    - include package_data for the medic rules, listed also in the pyproject
    Therefore, we in this script, we build the model package using most of spaCy's code
    and build a new pyproject.toml derived from the main one (at the root of the repo)
    and use poetry to build the model instead of `python setup.py`.

    SpaCy's original docstring:
    Generate an installable Python package for a pipeline. Includes binary data,
    meta and required installation files. A new directory will be created in the
    specified output directory, and the data will be copied over. If
    --create-meta is set and a meta.json already exists in the output directory,
    the existing values will be used as the defaults in the command-line prompt.
    After packaging, "python setup.py sdist" is run in the package directory,
    which will create a .tar.gz archive that can be installed via "pip install".
    If additional code files are provided (e.g. Python files containing custom
    registered functions like pipeline components), they are copied into the
    package and imported in the __init__.py.
    DOCS: https://spacy.io/api/cli#package
    """
    create_sdist, create_wheel = get_build_formats(string_to_list(build))
    code_paths = [Path(p.strip()) for p in string_to_list(code_paths)]
    package(
        input_dir,
        output_dir,
        meta_path=meta_path,
        code_paths=code_paths,
        name=name,
        version=version,
        create_meta=create_meta,
        create_sdist=create_sdist,
        create_wheel=create_wheel,
        force=force,
        silent=False,
    )


def package(
    input_dir: Path,
    output_dir: Path,
    meta_path: Optional[Path] = None,
    code_paths: List[Path] = [],
    name: Optional[str] = None,
    version: Optional[str] = None,
    create_meta: bool = False,
    create_sdist: bool = True,
    create_wheel: bool = False,
    force: bool = False,
    silent: bool = True,
) -> None:
    msg = Printer(no_print=silent, pretty=not silent)
    input_path = util.ensure_path(input_dir)
    output_path = util.ensure_path(output_dir)
    meta_path = util.ensure_path(meta_path)
    if create_wheel and not has_wheel():
        err = "Generating a binary .whl file requires wheel to be installed"
        msg.fail(err, "pip install wheel", exits=1)
    if not input_path or not input_path.exists():
        msg.fail("Can't locate pipeline data", input_path, exits=1)
    if not output_path or not output_path.exists():
        msg.fail("Output directory not found", output_path, exits=1)
    if create_sdist or create_wheel:
        opts = ["sdist" if create_sdist else "", "wheel" if create_wheel else ""]
        msg.info(f"Building package artifacts: {', '.join(opt for opt in opts if opt)}")
    for code_path in code_paths:
        if not code_path.exists():
            msg.fail("Can't find code file", code_path, exits=1)
        if os.path.isdir(code_path):
            print("Will import", code_path.stem, "but did not test it before packaging")
        # Import the code here so it's available when model is loaded (via
        # get_meta helper). Also verifies that everything works
        else:
            util.import_file(code_path.stem, code_path)
    if code_paths:
        msg.good(f"Including {len(code_paths)} Python module(s) with custom code")
    if meta_path and not meta_path.exists():
        msg.fail("Can't find pipeline meta.json", meta_path, exits=1)
    meta_path = meta_path or input_dir / "meta.json"
    if not meta_path.exists() or not meta_path.is_file():
        msg.fail("Can't load pipeline meta.json", meta_path, exits=1)
    meta = srsly.read_json(meta_path)
    meta = get_meta(input_dir, meta)
    if meta["requirements"]:
        msg.good(
            f"Including {len(meta['requirements'])} package requirement(s) from "
            f"meta and config",
            ", ".join(meta["requirements"]),
        )
    if name is not None:
        if not name.isidentifier():
            msg.fail(
                f"Model name ('{name}') is not a valid module name. "
                "This is required so it can be imported as a module.",
                "We recommend names that use ASCII A-Z, a-z, _ (underscore), "
                "and 0-9. "
                "For specific details see: "
                "https://docs.python.org/3/reference/lexical_analysis.html#identifiers",
                exits=1,
            )
        if not _is_permitted_package_name(name):
            msg.fail(
                f"Model name ('{name}') is not a permitted package name. "
                "This is required to correctly load the model with spacy.load.",
                "We recommend names that use ASCII A-Z, a-z, _ (underscore), "
                "and 0-9. "
                "For specific details see: "
                "https://www.python.org/dev/peps/pep-0426/#name",
                exits=1,
            )
        meta["name"] = name
    if version is not None:
        meta["version"] = version
    if not create_meta:  # only print if user doesn't want to overwrite
        msg.good("Loaded meta.json from file", meta_path)
    else:
        meta = generate_meta(meta, msg)
    errors = validate(ModelMetaSchema, meta)
    if errors:
        msg.fail("Invalid pipeline meta.json")
        print("\n".join(errors))
        sys.exit(1)
    model_name = meta["name"]
    if not model_name.startswith(meta["lang"] + "_"):
        model_name = f"{meta['lang']}_{model_name}"
    model_name_v = model_name + "-" + meta["version"]
    main_path = output_dir / model_name_v
    package_path = main_path / model_name
    if package_path.exists():
        if force:
            shutil.rmtree(str(package_path))
        else:
            msg.fail(
                "Package directory already exists",
                "Please delete the directory and try again, or use the "
                "`--force` flag to overwrite existing directories.",
                exits=1,
            )
    Path.mkdir(package_path, parents=True)
    shutil.copytree(str(input_dir), str(package_path / model_name_v))
    for file_name in FILENAMES_DOCS:
        file_path = package_path / model_name_v / file_name
        if file_path.exists():
            shutil.copy(str(file_path), str(main_path))
    readme_path = main_path / "README.md"
    if not readme_path.exists():
        readme = generate_readme(meta)
        create_file(readme_path, readme)
        create_file(package_path / model_name_v / "README.md", readme)
        msg.good("Generated README.md from meta.json")
    else:
        msg.info("Using existing README.md from pipeline directory")
    imports = []
    for code_path in code_paths:
        imports.append(code_path.stem)
        if os.path.isdir(code_path):
            print("Copying module", code_path, "to", str(package_path / code_path.stem))
            shutil.copytree(str(code_path), str(package_path / code_path.stem))
        else:
            shutil.copy(str(code_path), str(package_path))

    # no more top level meta.json, it was only used to load version
    # number and toplevel resources are not compatible with poetry
    create_file(main_path / model_name / "meta.json", srsly.json_dumps(meta, indent=2))

    # no more setup.py, we use poetry now
    # create_file(main_path / "setup.py", TEMPLATE_SETUP)

    create_file(
        main_path / "pyproject.toml",
        make_pyproject_toml(
            "pyproject.toml",
            model_name,
            model_name,
            model_name_v,
        ),
    )
    create_file(main_path / "MANIFEST.in", TEMPLATE_MANIFEST)
    init_py = TEMPLATE_INIT.format(
        imports="\n".join(f"from . import {m}" for m in imports),
        version=repr(version),
    )
    create_file(package_path / "__init__.py", init_py)
    msg.good(f"Successfully created package directory '{model_name_v}'", main_path)
    if create_sdist:
        with util.working_dir(main_path):
            util.run_command(["poetry", "build", "-f", "sdist"])
        zip_file = main_path / "dist" / f"{model_name_v}{SDIST_SUFFIX}"
        msg.good(f"Successfully created zipped Python package {zip_file}")
    if create_wheel:
        with util.working_dir(main_path):
            util.run_command(["poetry", "build", "-f", "wheel"])
        wheel_name_squashed = re.sub("_+", "_", model_name_v)
        wheel = main_path / "dist" / f"{wheel_name_squashed}{WHEEL_SUFFIX}"
        msg.good(f"Successfully created binary wheel {wheel}")
    if "__" in model_name:
        msg.warn(
            f"Model name ('{model_name}') contains a run of underscores. "
            "Runs of underscores are not significant in installed package names.",
        )


if __name__ == "__main__":
    app()
