
from typing import *
import torch
from . import _C
from . import pytorch


@torch.no_grad()
def encode(coords: torch.Tensor, permute: List[int] = [0, 1, 2], mode: Literal['z_order', 'hilbert'] = 'z_order') -> torch.Tensor:
    """
    Encodes 3D coordinates into a 30-bit code.

    Args:
        coords: a tensor of shape [N, 3] containing the 3D coordinates.
        permute: the permutation of the coordinates.
        mode: the encoding mode to use.
    """
    assert coords.shape[-1] == 3 and coords.ndim == 2, "Input coordinates must be of shape [N, 3]"
    x = coords[:, permute[0]].int()
    y = coords[:, permute[1]].int()
    z = coords[:, permute[2]].int()
    if mode == 'z_order':
        return _C.z_order_encode(x, y, z)
    elif mode == 'hilbert':
        return _C.hilbert_encode(x, y, z)
    else:
        raise ValueError(f"Unknown encoding mode: {mode}")


@torch.no_grad()
def decode(code: torch.Tensor, permute: List[int] = [0, 1, 2], mode: Literal['z_order', 'hilbert'] = 'z_order') -> torch.Tensor:
    """
    Decodes a 30-bit code into 3D coordinates.

    Args:
        code: a tensor of shape [N] containing the 30-bit code.
        permute: the permutation of the coordinates.
        mode: the decoding mode to use.
    """
    assert code.ndim == 1, "Input code must be of shape [N]"
    if mode == 'z_order':
        coords = _C.z_order_decode(code)
    elif mode == 'hilbert':
        coords = _C.hilbert_decode(code)
    else:
        raise ValueError(f"Unknown decoding mode: {mode}")
    x = coords[permute.index(0)]
    y = coords[permute.index(1)]
    z = coords[permute.index(2)]
    return torch.stack([x, y, z], dim=-1)
