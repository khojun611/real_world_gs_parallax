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
import torch.nn as nn
import numpy as np
import imageio
import os
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# 3D 방향 벡터를 Equirectangular 맵의 2D UV 좌표로 변환하는 헬퍼 함수
def dir_to_uv(dirs):
    """
    Converts 3D direction vectors to 2D UV coordinates for an equirectangular map.
    dirs: (..., 3) tensor of direction vectors
    Returns: (..., 2) tensor of UV coordinates in [0, 1] range
    """
    x, y, z = dirs.unbind(-1)
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    phi = torch.atan2(y, x)
    
    u = phi / (2 * torch.pi) + 0.5
    v = theta / torch.pi
    
    return torch.stack([u, v], dim=-1)

# CubemapEncoder 의존성을 제거한 새로운 EnvLight 클래스
class EnvLight(nn.Module):
    def __init__(self, path, device, max_res=128, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.max_res = max_res
        self.trainable = trainable
        
        try:
            self.brdf_lut = torch.from_numpy(np.fromfile('luts/lut_ggx_fg.bin', dtype=np.float32).reshape(1, 256, 256, 2)).to(device)
        except FileNotFoundError:
            print("Warning: 'luts/lut_ggx_fg.bin' not found. BRDF LUT will not be available.")
            self.brdf_lut = None

        if path is not None:
            self.load(path)
        else:
            # 학습 가능한 환경맵을 Equirectangular 형태로 생성
            self.base = nn.Parameter(torch.randn(1, 3, self.max_res, self.max_res * 2))
            if not self.trainable:
                self.base.requires_grad = False
        
        self.build_mips()

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDR image not found at: {path}")
            
        try:
            img = imageio.imread(path).astype(np.float32)
        except Exception as e:
            raise IOError(f"Could not read HDR image: {e}")

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
        self.base = nn.Parameter(img_tensor, requires_grad=self.trainable)

    def build_mips(self, num_levels=8):
        """
        torchvision의 gaussian_blur를 사용해 Mipmap을 직접 생성합니다.
        """
        # <<-- 핵심 수정: nn.ModuleList 대신 nn.ParameterList를 사용합니다. -->>
        mips_list = [self.base] # self.base는 이미 nn.Parameter입니다.
        base_map = self.base.data
        for i in range(1, num_levels):
            sigma = float(2**i) * 0.5
            kernel_size = int(sigma * 4) + 1
            if kernel_size % 2 == 0: kernel_size += 1
            
            blurred_map = gaussian_blur(base_map, kernel_size=kernel_size, sigma=sigma)
            mips_list.append(nn.Parameter(blurred_map, requires_grad=self.trainable))
        
        self.mips = nn.ParameterList(mips_list)

    def forward(self, x, roughness=None, mode="train"):
        # x: direction vectors (..., 3)
        
        if mode == "pure_env" or roughness is None:
            # Equirectangular 맵에서 직접 샘플링
            uv = dir_to_uv(x)
            # grid_sample을 위해 UV 좌표를 [-1, 1] 범위로 변환
            grid = (2.0 * uv - 1.0)
            
            # 입력 텐서의 차원에 맞게 grid 차원 조정
            if grid.ndim == 2: grid = grid.unsqueeze(0)
            if grid.ndim == 3: grid = grid.unsqueeze(0) # (B, H, W, 2) -> (B, 1, H, W, 2) for grid_sample
            
            sampled = F.grid_sample(self.mips[0], grid, mode='bilinear', padding_mode='border', align_corners=False)
            
            # 결과 텐서의 차원을 입력에 맞게 조정
            if x.ndim > 2:
                return sampled.squeeze(2).permute(0, 2, 3, 1).squeeze(0)
            else:
                 return sampled.squeeze()

        # 1. 모든 Mipmap 레벨에서 색상을 미리 샘플링
        uv = dir_to_uv(x)
        grid = (2.0 * uv - 1.0).unsqueeze(0) # (1, H, W, 2)
        
        sampled_colors = []
        for mip_map in self.mips:
            sampled = F.grid_sample(mip_map, grid, mode='bilinear', padding_mode='border', align_corners=False)
            sampled = sampled.squeeze(0).permute(1, 2, 0) # (H, W, 3)
            sampled_colors.append(sampled)
        
        all_mip_colors = torch.stack(sampled_colors, dim=0) # (num_mips, H, W, 3)
        
        # 2. Roughness 값에 따라 두 Mipmap 레벨 사이를 보간
        roughness = torch.clamp(roughness, self.min_roughness, self.max_roughness)
        idx = (roughness - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.mips) - 1)
        
        idx_floor = torch.clamp(idx.floor().long(), 0, len(self.mips)-2)
        idx_ceil = torch.clamp(idx.ceil().long(), 0, len(self.mips)-1)
        weight = (idx - idx.floor())
        
        # 3. 각 픽셀에 해당하는 Mipmap 색상을 가져오기 (gather 사용)
        M, H, W, C = all_mip_colors.shape
        all_mip_colors_flat = all_mip_colors.permute(1, 2, 0, 3).reshape(H * W, M, C)
        
        idx_floor_flat = idx_floor.reshape(H * W, 1, 1).expand(-1, -1, C)
        idx_ceil_flat = idx_ceil.reshape(H * W, 1, 1).expand(-1, -1, C)
        
        color_from = torch.gather(all_mip_colors_flat, 1, idx_floor_flat).squeeze(1).reshape(H, W, C)
        color_to = torch.gather(all_mip_colors_flat, 1, idx_ceil_flat).squeeze(1).reshape(H, W, C)
        
        # 4. 최종 색상 보간
        return torch.lerp(color_from, color_to, weight)
