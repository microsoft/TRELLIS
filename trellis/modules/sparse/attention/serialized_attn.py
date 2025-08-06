from typing import *
from enum import Enum
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
    'sparse_serialized_scaled_dot_product_self_attention',
]


class SerializeMode(Enum):
    Z_ORDER = 0
    Z_ORDER_TRANSPOSED = 1
    HILBERT = 2
    HILBERT_TRANSPOSED = 3


SerializeModes = [
    SerializeMode.Z_ORDER,
    SerializeMode.Z_ORDER_TRANSPOSED,
    SerializeMode.HILBERT,
    SerializeMode.HILBERT_TRANSPOSED
]


def calc_serialization(
    tensor: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        serialize_mode (SerializeMode): The serialization mode to use.
        shift_sequence (int): The shift of serialized sequence.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor, torch.Tensor): Forwards and backwards indices.
    """
    fwd_indices = []
    bwd_indices = []
    seq_lens = []
    seq_batch_indices = []
    offsets = [0]
    
    if 'vox2seq' not in globals():
        import vox2seq

    # Serialize the input
    serialize_coords = tensor.coords[:, 1:].clone()
    serialize_coords += torch.tensor(shift_window, dtype=torch.int32, device=tensor.device).reshape(1, 3)
    if serialize_mode == SerializeMode.Z_ORDER:
        code = vox2seq.encode(serialize_coords, mode='z_order', permute=[0, 1, 2])
    elif serialize_mode == SerializeMode.Z_ORDER_TRANSPOSED:
        code = vox2seq.encode(serialize_coords, mode='z_order', permute=[1, 0, 2])
    elif serialize_mode == SerializeMode.HILBERT:
        code = vox2seq.encode(serialize_coords, mode='hilbert', permute=[0, 1, 2])
    elif serialize_mode == SerializeMode.HILBERT_TRANSPOSED:
        code = vox2seq.encode(serialize_coords, mode='hilbert', permute=[1, 0, 2])
    else:
        raise ValueError(f"Unknown serialize mode: {serialize_mode}")
    
    for bi, s in enumerate(tensor.layout):
        num_points = s.stop - s.start
        num_windows = (num_points + window_size - 1) // window_size
        valid_window_size = num_points / num_windows
        to_ordered = torch.argsort(code[s.start:s.stop])
        if num_windows == 1:
            fwd_indices.append(to_ordered)
            bwd_indices.append(torch.zeros_like(to_ordered).scatter_(0, to_ordered, torch.arange(num_points, device=tensor.device)))
            fwd_indices[-1] += s.start
            bwd_indices[-1] += offsets[-1]
            seq_lens.append(num_points)
            seq_batch_indices.append(bi)
            offsets.append(offsets[-1] + seq_lens[-1])
        else:
            # Partition the input
            offset = 0
            mids = [(i + 0.5) * valid_window_size + shift_sequence for i in range(num_windows)]
            split = [math.floor(i * valid_window_size + shift_sequence) for i in range(num_windows + 1)]
            bwd_index = torch.zeros((num_points,), dtype=torch.int64, device=tensor.device)
            for i in range(num_windows):
                mid = mids[i]
                valid_start = split[i]
                valid_end = split[i + 1]
                padded_start = math.floor(mid - 0.5 * window_size)
                padded_end = padded_start + window_size
                fwd_indices.append(to_ordered[torch.arange(padded_start, padded_end, device=tensor.device) % num_points])
                offset += valid_start - padded_start
                bwd_index.scatter_(0, fwd_indices[-1][valid_start-padded_start:valid_end-padded_start], torch.arange(offset, offset + valid_end - valid_start, device=tensor.device))
                offset += padded_end - valid_start
                fwd_indices[-1] += s.start
            seq_lens.extend([window_size] * num_windows)
            seq_batch_indices.extend([bi] * num_windows)
            bwd_indices.append(bwd_index + offsets[-1])
            offsets.append(offsets[-1] + num_windows * window_size)

    fwd_indices = torch.cat(fwd_indices)
    bwd_indices = torch.cat(bwd_indices)

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices
    

def sparse_serialized_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply serialized scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        serialize_mode (SerializeMode): The serialization mode to use.
        shift_sequence (int): The shift of serialized sequence.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        shift (int): The shift to use.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'serialization_{serialize_mode}_{window_size}_{shift_sequence}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_serialization(qkv, window_size, serialize_mode, shift_sequence, shift_window)
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
            assert (qkv_coords[start:start+seq_lens[i], 0] == seq_batch_indices[i]).all(), f"SparseWindowedScaledDotProductSelfAttention: batch index mismatch"
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
