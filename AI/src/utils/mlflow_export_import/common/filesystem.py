"""
Filesystem utilities - Local
"""
import os
import shutil

__all__ = [
    "LocalFileSystem",
    "exists",
    "mk_local_path",
    "get_filesystem"
]


class LocalFileSystem(object):
    @staticmethod
    def cp(src: str, dst: str) -> None:
        shutil.copytree(mk_local_path(src), mk_local_path(dst))

    @staticmethod
    def rm(path: str) -> None:
        shutil.rmtree(mk_local_path(path))

    @staticmethod
    def mkdirs(path: str) -> None:
        os.makedirs(mk_local_path(path), exist_ok=True)

    @staticmethod
    def write(path: str, content: str) -> None:
        with open(mk_local_path(path), "w", encoding="utf-8") as f:
            f.write(content)
########################################################################################################################


def mk_local_path(path: str) -> str:
    return path.replace("dbfs:", "/dbfs")


def exists(path: str) -> None:
    os.path.exists(mk_local_path(path))
########################################################################################################################


def get_filesystem(fpath: str) -> LocalFileSystem:
    """ Return the filesystem object matching the directory path. """
    if fpath.startswith("dbfs:"):
        raise ValueError("Databricks file system is not supported.")
    return LocalFileSystem()
