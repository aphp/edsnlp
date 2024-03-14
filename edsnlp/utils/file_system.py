import fnmatch
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pyarrow.fs
from fsspec import AbstractFileSystem
from fsspec import __version__ as fsspec_version
from fsspec.implementations.arrow import ArrowFSWrapper

FileSystem = Union[AbstractFileSystem, pyarrow.fs.FileSystem]

if fsspec_version < "2023.3.0":
    # Ugly hack to make fsspec's arrow implementation work in python 3.7
    # since arrow requires files to be seekable, and the default fsspec
    # open(..., seekable) parameter is False
    # See https://github.com/fsspec/filesystem_spec/pull/1186
    ArrowFSWrapper._open.__wrapped__.__defaults__ = ("rb", None, True)


def walk_match(
    fs: FileSystem,
    root: str,
    file_pattern: str,
    recursive: bool = True,
) -> list:
    if fsspec_version >= "2023.10.0":
        # Version fixes fsspec glob https://github.com/fsspec/filesystem_spec/pull/1329
        glob_str = os.path.join(root, "**" if recursive else "", file_pattern)
        return fs.glob(glob_str)
    return [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in fs.walk(
            root,
            maxdepth=None if recursive else 1,
        )
        for f in fnmatch.filter(files, file_pattern)
    ]


def normalize_fs_path(
    filesystem: Optional[FileSystem],
    path: Union[str, Path],
) -> Tuple[AbstractFileSystem, str]:
    path = str(path)

    if filesystem is None or (isinstance(path, str) and "://" in path):
        path = (
            os.path.abspath(path)
            if isinstance(path, Path) or "://" in path
            else f"file://{os.path.abspath(path)}"
        )
        inferred_fs, fs_path = pyarrow.fs.FileSystem.from_uri(path)
        filesystem = filesystem or inferred_fs
        assert inferred_fs.type_name == filesystem.type_name, (
            f"Protocol {inferred_fs.type_name} in path does not match "
            f"filesystem {filesystem.type_name}"
        )
        path = fs_path

    return (
        ArrowFSWrapper(filesystem)
        if isinstance(filesystem, pyarrow.fs.FileSystem)
        else filesystem
    ), path
