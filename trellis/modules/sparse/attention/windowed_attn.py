from typing import *
import torch
import math
from .. import SparseTensor
from .. import DEBUG, ATTN

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_windowed_scaled_dot_product_self_attention',
]


def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (List[int]): Sequence lengths.
        (List[int]): Sequence batch indices.
    """
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.int32) // OFFSET[0]
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    seq_batch_indices = seq_batch_indices[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices
    

def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        shift (int): The shift to use.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'window_partition_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, seq_batch_indices))
    else:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = serialization_spatial_cache

    M = fwd_indices.shape[0]
    T = qkv.feats.shape[0]
    H = qkv.feats.shape[2]
    C = qkv.feats.shape[3]
    
    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]

    if DEBUG:
        start = 0
        qkv_coords = qkv.coords[fwd_indices]
        for i in range(len(seq_lens)):
            seq_coords = qkv_coords[start:start+seq_lens[i]]
            assert (seq_coords[:, 0] == seq_batch_indices[i]).all(), f"SparseWindowedScaledDotProductSelfAttention: batch index mismatch"
            assert (seq_coords[:, 1:].max(dim=0).values - seq_coords[:, 1:].min(dim=0).values < window_size).all(), \
                    f"SparseWindowedScaledDotProductSelfAttention: window size exceeded"
            start += seq_lens[i]

    if all([seq_len == window_size for seq_len in seq_lens]):
        B = len(seq_lens)
        N = window_size
        qkv_feats = qkv_feats.reshape(B, N, 3, H, C)
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=2)                       # [B, N, H, C]
            out = xops.memory_efficient_attention(q, k, v)          # [B, N, H, C]
        elif ATTN == 'flash_attn':
            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)   # [B, N, H, C]
        else:
            raise ValueError(f"Unknown attention module: {ATTN}")
        out = out.reshape(B * N, H, C)                              # [M, H, C]
    else:
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=1)                       # [M, H, C]
            q = q.unsqueeze(0)                                      # [1, M, H, C]
            k = k.unsqueeze(0)                                      # [1, M, H, C]
            v = v.unsqueeze(0)                                      # [1, M, H, C]
            mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(seq_lens)
            out = xops.memory_efficient_attention(q, k, v, mask)[0] # [M, H, C]
        elif ATTN == 'flash_attn':
            cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)], dim=0) \
                        .to(qkv.device).int()
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, max(seq_lens)) # [M, H, C]

    out = out[bwd_indices]      # [T, H, C]

    if DEBUG:
        qkv_coords = qkv_coords[bwd_indices]
        assert torch.equal(qkv_coords, qkv.coords), "SparseWindowedScaledDotProductSelfAttention: coordinate mismatch"

    return qkv.replace(out)
