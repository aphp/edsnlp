from __future__ import annotations

import subprocess
from pathlib import Path

__version__ = "0.21.0"


def get_version(base_version: str = __version__) -> str:
    repo_root = next(
        (
            current
            for current in (
                Path(__file__).resolve().parent,
                *Path(__file__).resolve().parent.parents,
            )
            if (current / ".git").exists()
        ),
        None,
    )
    if repo_root is None:
        return base_version

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return base_version

    return (
        base_version
        if tag in {base_version, f"v{base_version}"}
        else f"{base_version}.dev0+g{commit_hash}"
    )


__version__ = get_version()
