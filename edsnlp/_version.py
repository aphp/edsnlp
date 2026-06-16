from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path

_BASE_VERSION = "0.21.0"


def get_version(base_version: str = _BASE_VERSION) -> str:
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / ".git").exists():  # pragma: nocover
        try:
            return metadata.version("edsnlp")
        except metadata.PackageNotFoundError:
            pass
        return base_version

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: nocover
        return base_version

    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        tag = None

    return (
        base_version
        if tag in {base_version, f"v{base_version}"}
        else f"{base_version}.dev0+g{commit_hash}"
    )


__version__ = get_version()
