import tempfile

import torch

__all__ = ["pack_hook", "unpack_hook"]


def pack_hook(tensor: torch.Tensor) -> tempfile.NamedTemporaryFile:
    tmp_file: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(delete=True)
    torch.save(tensor, tmp_file.name)
    return tmp_file


def unpack_hook(tmp_file: tempfile.NamedTemporaryFile) -> torch.Tensor:
    tensor: torch.Tensor = torch.load(tmp_file.name, weights_only=True)
    return tensor
