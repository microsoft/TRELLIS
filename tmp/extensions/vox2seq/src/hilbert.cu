#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

#include "hilbert.h"


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


__global__ void hilbert_encode_cuda(
    size_t N,
    const uint32_t* x,
    const uint32_t* y,
    const uint32_t* z,
    uint32_t* codes
) {
    size_t thread_id = cg::this_grid().thread_rank();
    if (thread_id >= N) return;

    uint32_t point[3] = {x[thread_id], y[thread_id], z[thread_id]};

    uint32_t m = 1 << 9, q, p, t;

    // Inverse undo excess work
    q = m;
    while (q > 1) {
        p = q - 1;
        for (int i = 0; i < 3; i++) {
            if (point[i] & q) {
                point[0] ^= p;  // invert
            } else {
                t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q >>= 1;
    }

    // Gray encode
    for (int i = 1; i < 3; i++) {
        point[i] ^= point[i - 1];
    }
    t = 0;
    q = m;
    while (q > 1) {
        if (point[2] & q) {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for (int i = 0; i < 3; i++) {
        point[i] ^= t;
    }

    // Convert to 3D Hilbert code
    uint32_t xx = expandBits(point[0]);
    uint32_t yy = expandBits(point[1]);
    uint32_t zz = expandBits(point[2]);

    codes[thread_id] = xx * 4 + yy * 2 + zz;
}


__global__ void hilbert_decode_cuda(
    size_t N,
    const uint32_t* codes,
    uint32_t* x,
    uint32_t* y,
    uint32_t* z
) {
    size_t thread_id = cg::this_grid().thread_rank();
    if (thread_id >= N) return;

    uint32_t point[3];
    point[0] = extractBits(codes[thread_id] >> 2);
    point[1] = extractBits(codes[thread_id] >> 1);
    point[2] = extractBits(codes[thread_id]);

    uint32_t m = 2 << 9, q, p, t;

    // Gray decode by H ^ (H/2)
    t = point[2] >> 1;
    for (int i = 2; i > 0; i--) {
        point[i] ^= point[i - 1];
    }
    point[0] ^= t;

    // Undo excess work
    q = 2;
    while (q != m) {
        p = q - 1;
        for (int i = 2; i >= 0; i--) {
            if (point[i] & q) {
                point[0] ^= p;
            } else {
                t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q <<= 1;
    }

    x[thread_id] = point[0];
    y[thread_id] = point[1];
    z[thread_id] = point[2];
}
