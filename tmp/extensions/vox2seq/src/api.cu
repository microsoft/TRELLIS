#include <torch/extension.h>
#include "api.h"
#include "z_order.h"
#include "hilbert.h"


torch::Tensor
z_order_encode(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z
) {
    // Allocate output tensor
    torch::Tensor codes = torch::empty_like(x);

    // Call CUDA kernel
    z_order_encode_cuda<<<(x.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        x.size(0),
        reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
        reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
        reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
        reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
    );

    return codes;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
z_order_decode(
    const torch::Tensor& codes
) {
    // Allocate output tensors
    torch::Tensor x = torch::empty_like(codes);
    torch::Tensor y = torch::empty_like(codes);
    torch::Tensor z = torch::empty_like(codes);

    // Call CUDA kernel
    z_order_decode_cuda<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        codes.size(0),
        reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
        reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(z.data_ptr<int>())
    );

    return std::make_tuple(x, y, z);
}


torch::Tensor
hilbert_encode(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z
) {
    // Allocate output tensor
    torch::Tensor codes = torch::empty_like(x);

    // Call CUDA kernel
    hilbert_encode_cuda<<<(x.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        x.size(0),
        reinterpret_cast<uint32_t*>(x.contiguous().data_ptr<int>()),
        reinterpret_cast<uint32_t*>(y.contiguous().data_ptr<int>()),
        reinterpret_cast<uint32_t*>(z.contiguous().data_ptr<int>()),
        reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
    );

    return codes;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
hilbert_decode(
    const torch::Tensor& codes
) {
    // Allocate output tensors
    torch::Tensor x = torch::empty_like(codes);
    torch::Tensor y = torch::empty_like(codes);
    torch::Tensor z = torch::empty_like(codes);

    // Call CUDA kernel
    hilbert_decode_cuda<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        codes.size(0),
        reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
        reinterpret_cast<uint32_t*>(x.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(y.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(z.data_ptr<int>())
    );

    return std::make_tuple(x, y, z);
}
