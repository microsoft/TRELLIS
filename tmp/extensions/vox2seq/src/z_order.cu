#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

#include "z_order.h"


// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
static __device__ uint32_t expandBits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}


// Removes 2 zeros after each bit in a 30-bit integer.
static __device__ uint32_t extractBits(uint32_t v)
{
    v = v & 0x49249249;
    v = (v ^ (v >>  2)) & 0x030C30C3u;
    v = (v ^ (v >>  4)) & 0x0300F00Fu;
    v = (v ^ (v >>  8)) & 0x030000FFu;
    v = (v ^ (v >> 16)) & 0x000003FFu;
    return v;
}


__global__ void z_order_encode_cuda(
    size_t N,
    const uint32_t* x,
    const uint32_t* y,
    const uint32_t* z,
    uint32_t* codes
) {
    size_t thread_id = cg::this_grid().thread_rank();
	if (thread_id >= N) return;

    uint32_t xx = expandBits(x[thread_id]);
    uint32_t yy = expandBits(y[thread_id]);
    uint32_t zz = expandBits(z[thread_id]);

    codes[thread_id] = xx * 4 + yy * 2 + zz;
}


__global__ void z_order_decode_cuda(
    size_t N,
    const uint32_t* codes,
    uint32_t* x,
    uint32_t* y,
    uint32_t* z
) {
    size_t thread_id = cg::this_grid().thread_rank();
    if (thread_id >= N) return;

    x[thread_id] = extractBits(codes[thread_id] >> 2);
    y[thread_id] = extractBits(codes[thread_id] >> 1);
    z[thread_id] = extractBits(codes[thread_id]);
}
