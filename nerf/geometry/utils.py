# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

def unbounded_unwarp(pts, radius=2):
    pts_norm = torch.norm(pts, dim=-1)
    scale = 1/(radius-pts_norm[..., None]) / pts_norm[..., None]
    mask_inside_inner_sphere = (pts_norm <= 1.0)[..., None]
    pts = torch.where(mask_inside_inner_sphere, pts, scale * pts)
    return pts

def unbounded_warp(pts, radius=2):
    pts_norm = torch.norm(pts, dim=-1)
    scale = (radius - 1.0 / pts_norm[..., None]) / pts_norm[..., None]
    mask_inside_inner_sphere = (pts_norm <= 1.0)[..., None]
    pts = torch.where(mask_inside_inner_sphere, pts, scale * pts)
    return pts