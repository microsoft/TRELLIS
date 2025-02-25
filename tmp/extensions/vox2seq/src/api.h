/*
 * Serialize a voxel grid
 *
 * Copyright (C) 2024, Jianfeng XIANG <belljig@outlook.com>
 * All rights reserved.
 *
 * Licensed under The MIT License [see LICENSE for details]
 *
 * Written by Jianfeng XIANG
 */

#pragma once
#include <torch/extension.h>


#define BLOCK_SIZE 256


/**
 * Z-order encode 3D points
 *
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 *
 * @return [N] tensor containing the z-order encoded values
 */
torch::Tensor
z_order_encode(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z
);


/**
 * Z-order decode 3D points
 *
 * @param codes [N] tensor containing the z-order encoded values
 *
 * @return 3 tensors [N] containing the x, y, z coordinates
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
z_order_decode(
    const torch::Tensor& codes
);


/**
 * Hilbert encode 3D points
 *
 * @param x [N] tensor containing the x coordinates
 * @param y [N] tensor containing the y coordinates
 * @param z [N] tensor containing the z coordinates
 *
 * @return [N] tensor containing the Hilbert encoded values
 */
torch::Tensor
hilbert_encode(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z
);


/**
 * Hilbert decode 3D points
 *
 * @param codes [N] tensor containing the Hilbert encoded values
 *
 * @return 3 tensors [N] containing the x, y, z coordinates
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
hilbert_decode(
    const torch::Tensor& codes
);
