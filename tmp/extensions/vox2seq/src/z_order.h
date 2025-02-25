#pragma once

/**
 * Z-order encode 3D points
 *
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 *
 * @return [N] tensor containing the z-order encoded values
 */
__global__ void z_order_encode_cuda(
    size_t N,
    const uint32_t* x,
    const uint32_t* y,
    const uint32_t* z,
    uint32_t* codes
);


/**
 * Z-order decode 3D points
 *
 * @param codes [N] tensor containing the z-order encoded values
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 */
__global__ void z_order_decode_cuda(
    size_t N,
    const uint32_t* codes,
    uint32_t* x,
    uint32_t* y,
    uint32_t* z
);
