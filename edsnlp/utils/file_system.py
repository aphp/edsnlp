import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import fsspec.implementations.local
import pyarrow.fs
from fsspec import AbstractFileSystem
from fsspec.implementations.arrow import ArrowFSWrapper

FileSystem = Union[AbstractFileSystem, pyarrow.fs.FileSystem]


def walk_match(
    fs: FileSystem,
    root: str,
    file_pattern: str,
) -> list:
    if fs.isfile(root):
        return [root]
    return [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in fs.walk(root)
        for f in files
        if re.match(file_pattern, f)
    ]


def normalize_fs_path(
    filesystem: Optional[FileSystem],
    path: Union[str, Path],
) -> Tuple[AbstractFileSystem, str]:
    has_protocol = isinstance(path, str) and "://" in path
    filesystem = (
        ArrowFSWrapper(filesystem)
        if isinstance(filesystem, pyarrow.fs.FileSystem)
        else filesystem
    )

    # We need to detect the fs from the path
    if filesystem is None or has_protocol:
        uri: str = path if has_protocol else f"file://{os.path.abspath(path)}"
        inferred_fs, fs_path = fsspec.core.url_to_fs(uri)
        inferred_fs: fsspec.AbstractFileSystem
        filesystem = filesystem or inferred_fs
        assert inferred_fs.protocol == filesystem.protocol, (
            f"Protocol {inferred_fs.protocol} in path does not match "
            f"filesystem {filesystem.protocol}"
        )
        path = fs_path  # path without protocol

    return (
        ArrowFSWrapper(filesystem)
        if isinstance(filesystem, pyarrow.fs.FileSystem)
        else filesystem
    ), str(path)
