#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="vox2seq",
    packages=['vox2seq', 'vox2seq.pytorch'],
    ext_modules=[
        CUDAExtension(
            name="vox2seq._C",
            sources=[
                "src/api.cu",
                "src/z_order.cu",
                "src/hilbert.cu",
                "src/ext.cpp",
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
