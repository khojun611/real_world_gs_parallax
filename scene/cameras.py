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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCorrect

class Camera(nn.Module):
    # --- << __init__ 함수의 인자 리스트를 수정합니다 >> ---
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
             image_name, uid, image_height, image_width,
             trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
             HWK=None, normal_map=None, has_normal=False, gt_refl_mask=None,
             pseudo_material_map=None, has_pseudo_material=False
             ):
        self.uid = uid
        self.data_device = data_device
        # ... (기존 self 변수 할당은 그대로) ...
        self.original_image = image
        self.gt_alpha_mask = gt_alpha_mask
        self.image_height = image_height
        self.image_width = image_width
        self.image_name = image_name
        self.FoVx = FoVx
        self.FoVy = FoVy
        
        self.normal_map = torch.from_numpy(normal_map).float().to(data_device) if has_normal else None
        self.has_normal = has_normal
        self.gt_refl_mask = gt_refl_mask.to(data_device) if gt_refl_mask is not None else None
        
        # --- << 새로운 self 변수를 할당합니다 >> ---
        self.pseudo_material_map = torch.from_numpy(pseudo_material_map).float().to(data_device) if has_pseudo_material else None
        self.has_pseudo_material = has_pseudo_material
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if HWK is None:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        else:
            self.HWK = HWK
            self.projection_matrix = getProjectionMatrixCorrect(self.znear, self.zfar, HWK[0], HWK[1], HWK[2]).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # --- 아래 두 줄이 누락되었습니다. 추가해주세요. ---
        self.R = torch.tensor(R, dtype=torch.float32, device=self.data_device)
        self.T = torch.tensor(T, dtype=torch.float32, device=self.data_device)
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

