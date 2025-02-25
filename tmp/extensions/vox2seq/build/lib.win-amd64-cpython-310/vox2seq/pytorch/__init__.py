import torch
from typing import *

from .default import (
    encode,
    decode,
    z_order_encode,
    z_order_decode,
    hilbert_encode,
    hilbert_decode,
)


@torch.no_grad()
def encode(coords: torch.Tensor, permute: List[int] = [0, 1, 2], mode: Literal['z_order', 'hilbert'] = 'z_order') -> torch.Tensor:
    """
    Encodes 3D coordinates into a 30-bit code.

    Args:
        coords: a tensor of shape [N, 3] containing the 3D coordinates.
        permute: the permutation of the coordinates.
        mode: the encoding mode to use.
    """
    if mode == 'z_order':
        return z_order_encode(coords[:, permute], depth=10).int()
    elif mode == 'hilbert':
        return hilbert_encode(coords[:, permute], depth=10).int()
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
    if mode == 'z_order':
        return z_order_decode(code, depth=10)[:, permute].float()
    elif mode == 'hilbert':
        return hilbert_decode(code, depth=10)[:, permute].float()
    else:
        raise ValueError(f"Unknown decoding mode: {mode}")
    