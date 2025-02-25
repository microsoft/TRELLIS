import time
import torch
import vox2seq


if __name__ == "__main__":
    stats = {
        'z_order_cuda': [],
        'z_order_pytorch': [],
        'hilbert_cuda': [],
        'hilbert_pytorch': [],
    }
    RES = [16, 32, 64, 128, 256]
    for res in RES:
        coords = torch.meshgrid(torch.arange(res), torch.arange(res), torch.arange(res))
        coords = torch.stack(coords, dim=-1).reshape(-1, 3).int().cuda()

        start = time.time()
        for _ in range(100):
            code_z_cuda = vox2seq.encode(coords, mode='z_order').cuda()
        torch.cuda.synchronize()
        stats['z_order_cuda'].append((time.time() - start) / 100)

        start = time.time()
        for _ in range(100):
            code_z_pytorch = vox2seq.pytorch.encode(coords, mode='z_order').cuda()
        torch.cuda.synchronize()
        stats['z_order_pytorch'].append((time.time() - start) / 100)

        start = time.time()
        for _ in range(100):
            code_h_cuda = vox2seq.encode(coords, mode='hilbert').cuda()
        torch.cuda.synchronize()
        stats['hilbert_cuda'].append((time.time() - start) / 100)

        start = time.time()
        for _ in range(100):
            code_h_pytorch = vox2seq.pytorch.encode(coords, mode='hilbert').cuda()
        torch.cuda.synchronize()
        stats['hilbert_pytorch'].append((time.time() - start) / 100)

    print(f"{'Resolution':<12}{'Z-Order (CUDA)':<24}{'Z-Order (PyTorch)':<24}{'Hilbert (CUDA)':<24}{'Hilbert (PyTorch)':<24}")
    for res, z_order_cuda, z_order_pytorch, hilbert_cuda, hilbert_pytorch in zip(RES, stats['z_order_cuda'], stats['z_order_pytorch'], stats['hilbert_cuda'], stats['hilbert_pytorch']):
        print(f"{res:<12}{z_order_cuda:<24.6f}{z_order_pytorch:<24.6f}{hilbert_cuda:<24.6f}{hilbert_pytorch:<24.6f}")

