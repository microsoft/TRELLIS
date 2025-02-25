import torch
import vox2seq


if __name__ == "__main__":
    RES = 256
    coords = torch.meshgrid(torch.arange(RES), torch.arange(RES), torch.arange(RES))
    coords = torch.stack(coords, dim=-1).reshape(-1, 3).int().cuda()
    code_z_cuda = vox2seq.encode(coords, mode='z_order')
    code_z_pytorch = vox2seq.pytorch.encode(coords, mode='z_order')
    code_h_cuda = vox2seq.encode(coords, mode='hilbert')
    code_h_pytorch = vox2seq.pytorch.encode(coords, mode='hilbert')
    assert torch.equal(code_z_cuda, code_z_pytorch)
    assert torch.equal(code_h_cuda, code_h_pytorch)

    code = torch.arange(RES**3).int().cuda()
    coords_z_cuda = vox2seq.decode(code, mode='z_order')
    coords_z_pytorch = vox2seq.pytorch.decode(code, mode='z_order')
    coords_h_cuda = vox2seq.decode(code, mode='hilbert')
    coords_h_pytorch = vox2seq.pytorch.decode(code, mode='hilbert')
    assert torch.equal(coords_z_cuda, coords_z_pytorch)
    assert torch.equal(coords_h_cuda, coords_h_pytorch)

    print("All tests passed.")

