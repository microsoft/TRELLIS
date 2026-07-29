from typing import *
import torch
import torch.nn as nn
from . import BACKEND, DEBUG
SparseTensorData = None # Lazy import


def _load_sparse_tensor_data():
    global SparseTensorData
    if SparseTensorData is None:
        import importlib

        if BACKEND == 'torchsparse':
            SparseTensorData = importlib.import_module('torchsparse').SparseTensor
        elif BACKEND == 'spconv':
            SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor


__all__ = [
    'SparseTensor',
    'sparse_batch_broadcast',
    'sparse_batch_op',
    'sparse_cat',
    'sparse_unbind',
]


class SparseTensor:
    """
    Sparse tensor with support for both torchsparse and spconv backends.
    
    Parameters:
    - feats (torch.Tensor): Features of the sparse tensor.
    - coords (torch.Tensor): Coordinates of the sparse tensor.
    - shape (torch.Size): Shape of the sparse tensor.
    - layout (List[slice]): Layout of the sparse tensor for each batch
    - data (SparseTensorData): Sparse tensor data used for convolusion

    NOTE:
    - Data corresponding to a same batch should be contiguous.
    - Coords should be in [0, 1023]
    """
    @overload
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor, shape: Optional[torch.Size] = None, layout: Optional[List[slice]] = None, **kwargs): ...

    @overload
    def __init__(self, data, shape: Optional[torch.Size] = None, layout: Optional[List[slice]] = None, **kwargs): ...

    def __init__(self, *args, **kwargs):
        # Lazy import of sparse tensor backend
        _load_sparse_tensor_data()
                
        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if 'data' in kwargs else 0

        if method_id == 0:
            feats, coords, shape, layout = args + (None,) * (4 - len(args))
            if 'feats' in kwargs:
                feats = kwargs['feats']
                del kwargs['feats']
            if 'coords' in kwargs:
                coords = kwargs['coords']
                del kwargs['coords']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']
            if 'layout' in kwargs:
                layout = kwargs['layout']
                del kwargs['layout']

            if shape is None:
                shape = self.__cal_shape(feats, coords)
            if layout is None:
                layout = self.__cal_layout(coords, shape[0])
            if BACKEND == 'torchsparse':
                self.data = SparseTensorData(feats, coords, **kwargs)
            elif BACKEND == 'spconv':
                spatial_shape = list(coords.max(0)[0] + 1)[1:]
                self.data = SparseTensorData(feats.reshape(feats.shape[0], -1), coords, spatial_shape, shape[0], **kwargs)
                self.data._features = feats
        elif method_id == 1:
            data, shape, layout = args + (None,) * (3 - len(args))
            if 'data' in kwargs:
                data = kwargs['data']
                del kwargs['data']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']
            if 'layout' in kwargs:
                layout = kwargs['layout']
                del kwargs['layout']

            self.data = data
            if shape is None:
                shape = self.__cal_shape(self.feats, self.coords)
            if layout is None:
                layout = self.__cal_layout(self.coords, shape[0])

        self._shape = shape
        self._layout = layout
        self._scale = kwargs.get('scale', (1, 1, 1))
        self._spatial_cache = kwargs.get('spatial_cache', {})

        if DEBUG:
            try:
                assert self.feats.shape[0] == self.coords.shape[0], f"Invalid feats shape: {self.feats.shape}, coords shape: {self.coords.shape}"
                assert self.shape == self.__cal_shape(self.feats, self.coords), f"Invalid shape: {self.shape}"
                assert self.layout == self.__cal_layout(self.coords, self.shape[0]), f"Invalid layout: {self.layout}"
                for i in range(self.shape[0]):
                    assert torch.all(self.coords[self.layout[i], 0] == i), f"The data of batch {i} is not contiguous"
            except Exception as e:
                print('Debugging information:')
                print(f"- Shape: {self.shape}")
                print(f"- Layout: {self.layout}")
                print(f"- Scale: {self._scale}")
                print(f"- Coords: {self.coords}")
                raise e
        
    def __cal_shape(self, feats, coords):
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)
    
    def __cal_layout(self, coords, batch_size):
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0) 
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]
        return layout
    
    @property
    def shape(self) -> torch.Size:
        return self._shape
    
    def dim(self) -> int:
        return len(self.shape)
    
    @property
    def layout(self) -> List[slice]:
        return self._layout

    @property
    def feats(self) -> torch.Tensor:
        if BACKEND == 'torchsparse':
            return self.data.F
        elif BACKEND == 'spconv':
            return self.data.features
    
    @feats.setter
    def feats(self, value: torch.Tensor):
        if BACKEND == 'torchsparse':
            self.data.F = value
        elif BACKEND == 'spconv':
            self.data.features = value

    @property
    def coords(self) -> torch.Tensor:
        if BACKEND == 'torchsparse':
            return self.data.C
        elif BACKEND == 'spconv':
            return self.data.indices
        
    @coords.setter
    def coords(self, value: torch.Tensor):
        if BACKEND == 'torchsparse':
            self.data.C = value
        elif BACKEND == 'spconv':
            self.data.indices = value

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    @overload
    def to(self, dtype: torch.dtype) -> 'SparseTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> 'SparseTensor': ...

    def to(self, *args, **kwargs) -> 'SparseTensor':
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']
        
        new_feats = self.feats.to(device=device, dtype=dtype)
        new_coords = self.coords.to(device=device)
        return self.replace(new_feats, new_coords)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'SparseTensor':
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        return self.replace(new_feats, new_coords)
    
    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)

    def half(self) -> 'SparseTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)
    
    def float(self) -> 'SparseTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)
    
    def detach(self) -> 'SparseTensor':
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        return self.replace(new_feats, new_coords)

    def dense(self) -> torch.Tensor:
        if BACKEND == 'torchsparse':
            return self.data.dense()
        elif BACKEND == 'spconv':
            return self.data.dense()

    def reshape(self, *shape) -> 'SparseTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)
    
    def unbind(self, dim: int) -> List['SparseTensor']:
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> 'SparseTensor':
        _load_sparse_tensor_data()
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])
        if BACKEND == 'torchsparse':
            new_data = SparseTensorData(
                feats=feats,
                coords=self.data.coords if coords is None else coords,
                stride=self.data.stride,
                spatial_range=self.data.spatial_range,
            )
            new_data._caches = self.data._caches
        elif BACKEND == 'spconv':
            new_data = SparseTensorData(
                self.data.features.reshape(self.data.features.shape[0], -1),
                self.data.indices,
                self.data.spatial_shape,
                self.data.batch_size,
                self.data.grid,
                self.data.voxel_num,
                self.data.indice_dict
            )
            new_data._features = feats
            new_data.benchmark = self.data.benchmark
            new_data.benchmark_record = self.data.benchmark_record
            new_data.thrust_allocator = self.data.thrust_allocator
            new_data._timer = self.data._timer
            new_data.force_algo = self.data.force_algo
            new_data.int8_scale = self.data.int8_scale
            if coords is not None:
                new_data.indices = coords
        new_tensor = SparseTensor(new_data, shape=torch.Size(new_shape), layout=self.layout, scale=self._scale, spatial_cache=self._spatial_cache)
        return new_tensor

    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> 'SparseTensor':
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        coords = torch.cat([
            torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
            coords.repeat(N, 1),
        ], dim=1).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats=feats, coords=coords)

    def __merge_sparse_cache(self, other: 'SparseTensor') -> dict:
        new_cache = {}
        for k in set(list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k]
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k]
                else:
                    new_cache[k].update(other._spatial_cache[k])
        return new_cache

    def __neg__(self) -> 'SparseTensor':
        return self.replace(-self.feats)
    
    def __elemwise__(self, other: Union[torch.Tensor, 'SparseTensor'], op: callable) -> 'SparseTensor':
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = sparse_batch_broadcast(self, other)
            except:
                pass
        if isinstance(other, SparseTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor

    def __add__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.add)
    
    def __sub__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.sub)
    
    def __rsub__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")
        
        coords = []
        feats = []
        for new_idx, old_idx in enumerate(idx):
            coords.append(self.coords[self.layout[old_idx]].clone())
            coords[-1][:, 0] = new_idx
            feats.append(self.feats[self.layout[old_idx]])
        coords = torch.cat(coords, dim=0).contiguous()
        feats = torch.cat(feats, dim=0).contiguous()
        return SparseTensor(feats=feats, coords=coords)

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.
    
    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    coords, feats = input.coords, input.feats
    broadcasted = torch.zeros_like(feats)
    for k in range(input.shape[0]):
        broadcasted[input.layout[k]] = other[k]
    return broadcasted


def sparse_batch_op(input: SparseTensor, other: torch.Tensor, op: callable = torch.add) -> SparseTensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.
    
    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    return input.replace(op(input.feats, sparse_batch_broadcast(input, other)))


def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    """
    Concatenate a list of sparse tensors.
    
    Args:
        inputs (List[SparseTensor]): List of sparse tensors to concatenate.
    """
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    """
    Unbind a sparse tensor along a dimension.
    
    Args:
        input (SparseTensor): Sparse tensor to unbind.
        dim (int): Dimension to unbind.
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]
