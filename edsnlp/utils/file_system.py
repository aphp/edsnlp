import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pyarrow.fs
from fsspec import AbstractFileSystem
from fsspec.implementations.arrow import ArrowFSWrapper

FileSystem = Union[AbstractFileSystem, pyarrow.fs.FileSystem]


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
