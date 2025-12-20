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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import spatial_gradient
from .image_utils import psnr
from utils.image_utils import erode
import numpy as np
import torchvision.transforms.functional as TF


def rgb_to_luminance(rgb_tensor):
    # BT.709 표준에 따른 휘도 계산
    # rgb_tensor shape: (C, H, W)
    return 0.2126 * rgb_tensor[0:1, :, :] + 0.7152 * rgb_tensor[1:2, :, :] + 0.0722 * rgb_tensor[2:3, :, :]

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_high_frequency_mask(image):
    grayscale_image = torch.mean(image, dim=0, keepdim=True).unsqueeze(0)
    grads = spatial_gradient(grayscale_image, order=1)
    grad_x, grad_y = grads[:, :, 0, :, :], grads[:, :, 1, :, :]
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    min_val, max_val = torch.min(grad_magnitude), torch.max(grad_magnitude)
    if max_val > min_val:
        grad_magnitude = (grad_magnitude - min_val) / (max_val - min_val)
    return grad_magnitude.squeeze(0)


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def calculate_purity(rgb_image, epsilon=1e-8):
    """
    RGB 이미지의 색 순도를 계산합니다.
    순도 = max(R,G,B) / (R+G+B)
    """
    # rgb_image shape: [C, H, W] (C=3)
    # 채널이 3개가 아니거나 비어있으면 0을 반환하여 오류 방지
    if rgb_image.shape[0] != 3 or rgb_image.numel() == 0:
        return torch.zeros_like(rgb_image[:1])

    s = torch.sum(rgb_image, dim=0, keepdim=True) + epsilon
    m, _ = torch.max(rgb_image, dim=0, keepdim=True)
    
    purity = m / s
    return purity


def calculate_chromaticity(rgb_image, epsilon=1e-8):
    """
    RGB 이미지 또는 벡터의 크로마티시티를 계산합니다.
    Chromaticity = (R,G,B) / (R+G+B)
    """
    # 입력 텐서의 차원(ndim)을 확인하여 처리 방식을 분기합니다.
    if rgb_image.ndim >= 3:  # 이미지 텐서 (e.g., [C, H, W])
        channel_dim = -3
        if rgb_image.shape[channel_dim] != 3 or rgb_image.numel() == 0:
            return torch.zeros_like(rgb_image[..., :1, :, :])
    elif rgb_image.ndim == 1: # 색상 벡터 (e.g., [3])
        channel_dim = 0
        if rgb_image.shape[channel_dim] != 3 or rgb_image.numel() == 0:
            return torch.zeros_like(rgb_image)
    else: # 그 외의 경우 안전하게 0을 반환
        return torch.zeros_like(rgb_image)

    s = torch.sum(rgb_image, dim=channel_dim, keepdim=True) + epsilon
    chroma = rgb_image / s
    return chroma


def create_fourier_high_frequency_mask(image_tensor, cutoff_radius_ratio=0.1):
    """
    푸리에 변환을 이용해 이미지의 고주파 영역을 강조하는 마스크를 생성합니다.
    """
    # 1. 이미지 전처리 (흑백 변환)
    grayscale_image = TF.rgb_to_grayscale(image_tensor).squeeze(0) # [H, W]
    
    # 2. 2D 푸리에 변환
    fft_result = torch.fft.fft2(grayscale_image)
    shifted_fft = torch.fft.fftshift(fft_result)

    # 3. 고주파 통과 필터(High-Pass Filter) 생성
    H, W = grayscale_image.shape
    center_h, center_w = H // 2, W // 2
    
    y, x = torch.meshgrid(torch.arange(H, device=image_tensor.device), 
                          torch.arange(W, device=image_tensor.device), 
                          indexing='ij')
    distance_from_center = torch.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    max_radius = min(center_h, center_w)
    cutoff_radius = max_radius * cutoff_radius_ratio
    
    high_pass_filter = (distance_from_center > cutoff_radius).float()

    # 4. 필터 적용 및 역변환
    filtered_spectrum = shifted_fft * high_pass_filter
    shifted_ifft = torch.fft.ifftshift(filtered_spectrum)
    inverse_fft_result = torch.fft.ifft2(shifted_ifft)

    # 5. 최종 마스크 생성
    high_freq_map = torch.abs(inverse_fft_result)
    
    min_val, max_val = high_freq_map.min(), high_freq_map.max()
    if max_val > min_val:
        high_freq_mask = (high_freq_map - min_val) / (max_val - min_val)
    else:
        high_freq_mask = torch.zeros_like(high_freq_map)
        
    return high_freq_mask.unsqueeze(0) # (1, H, W) 형태로 반환

# utils/loss_utils.py 파일 내

# utils/loss_utils.py 파일

# ... (l1_loss, ssim, calculate_purity 등 다른 헬퍼 함수들은 그대로 둡니다) ...

def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    
    #print("diffuse",render_pkg.get("diffuse_map").shape)
    #print("specular",render_pkg.get("specular_map").shape)
    
    # print(render_pkg.keys())
    #print(f"[INFO] In calculate_loss for '{viewpoint_camera.image_name}': has_material_map={viewpoint_camera.has_pseudo_material}")


    
    # 1. 렌더링된 가우시안이 없는 경우 예외 처리
    if render_pkg["visibility_filter"].sum() == 0:
        image = render_pkg["render"]
        gt_image = viewpoint_camera.original_image.cuda()
        loss = l1_loss(image, gt_image)
        tb_dict = {
            "loss_dist": 0.0, "loss_normal_render_depth": 0.0, 
            "loss0": loss.item(), "loss_l1": loss.item(), 
            "loss_normal_smooth": 0.0, "loss_depth_smooth": 0.0,
            "loss": loss.item(), "psnr": 0.0, "ssim": 0.0,
            "num_points": pc.get_xyz.shape[0], "loss_hint": 0.0,
            "loss_purity": 0.0, "loss_diffuse_cons": 0.0
        }
        return loss, tb_dict

    # --- 렌더링 결과물 언패킹 ---
    tb_dict = { "num_points": pc.get_xyz.shape[0] }
    rendered_image = render_pkg["render"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()
    uncertainty_map = render_pkg.get("rgb_uncertainty_map")
    

    # 2. 학습 단계에 따라 주 손실(image loss)을 동적으로 선택
    # 불확실성 맵이 비어있지 않은지 한번 더 검사
    if opt.uncertainty_from_iter > 0 and iteration > opt.uncertainty_from_iter and uncertainty_map is not None and uncertainty_map.numel() > 0:
        # 하이브리드 손실 (L1 + NLL) 계산
        uncertainty_map = uncertainty_map + 1e-8
        l2_error = (rendered_image - gt_image) ** 2
        nll_loss_map = 0.5 * (l2_error / uncertainty_map**2 + torch.log(uncertainty_map**2))
        #print("ucmap",uncertainty_map.shape)
        Lnll = nll_loss_map.mean()
        Ll1 = l1_loss(rendered_image, gt_image)
        image_loss = (1.0 - opt.lambda_hybrid) * Ll1 + opt.lambda_hybrid * Lnll
        tb_dict["loss_nll"] = Lnll.item()
        tb_dict["loss_l1"] = Ll1.item()
        
    else:
        # [수정] 일반 L1 대신 'Metallic Weighted L1' 적용
        # 1. 픽셀별 오차 계산 (평균 내지 않음)
        l1_diff = torch.abs(rendered_image - gt_image)
        
        # 2. 가중치 맵 생성
        # 기본 가중치 1.0, 금속 영역은 +10.0 (총 11.0배 중요하게 취급)
        pixel_weight = 1.0
        
        # [수정] 속성이 존재하고(hasattr), 그 값이 None이 아닐 때만(is not None) 실행
        if hasattr(viewpoint_camera, "pseudo_material_map") and viewpoint_camera.pseudo_material_map is not None:
            # [수정] 마스크 데이터 가져오기
            raw_mask = viewpoint_camera.pseudo_material_map
            
            # [핵심 수정] 차원(ndim)을 확인하여 안전하게 (1, H, W)로 변환
            if raw_mask.ndim == 2:
                # (H, W) -> (1, H, W) : 차원 추가
                metal_mask = raw_mask.unsqueeze(0).cuda()
            elif raw_mask.shape[0] == 1:
                # (1, H, W) : 그대로 사용
                metal_mask = raw_mask.cuda()
            else:
                # (3, H, W) 등 다채널인 경우 : 사용자의 원래 의도대로 1번 채널 사용
                # (만약 RGB 마스크라면 G채널, 아니라면 0:1로 수정 고려)
                metal_mask = raw_mask[1:2, :, :].cuda()
            
            # 금속 영역에 가중치 10배 추가 (조절 가능)
            metal_weight_scale = 3.0
            
            # 브로드캐스팅을 위해 metal_mask가 (1, H, W)인지 확인
            pixel_weight = 1.0 + (metal_mask * metal_weight_scale)
            
        # 3. 가중 평균 Loss 계산
        image_loss = (l1_diff * pixel_weight).mean()
        tb_dict["loss_l1"] = image_loss.item()
    
    # --- 3. 주 손실과 SSIM 손실 조합 ---
    ssim_val = ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
    loss0 = (1.0 - opt.lambda_dssim) * image_loss + opt.lambda_dssim * (1.0 - ssim_val)
    loss = loss0.clone()

    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    
    
    
    

    # --- 4. 모든 정규화(Regularization) 손실들 순서대로 추가 ---
    if opt.lambda_diffuse_cons > 0 and iteration > opt.diffuse_cons_from_iter:
        sh_rest = pc._features_rest
        loss_diffuse_cons = torch.mean(sh_rest**2)
        tb_dict["loss_diffuse_cons"] = loss_diffuse_cons.item()
        loss += opt.lambda_diffuse_cons * loss_diffuse_cons
    else:
        tb_dict["loss_diffuse_cons"] = 0.0
    # 법선 일관성 손실
    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None].mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss += opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = 0.0

    # 왜곡 손실
    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss.item()
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = 0.0

    # 법선 평활화 손실
    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_smooth_from_iter and iteration < opt.normal_smooth_until_iter:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss += opt.lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = 0.0
    
    # 깊이 평활화 손실
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        loss += opt.lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = 0.0
        
    # 힌트 손실 (Hint Loss)
    if opt.lambda_hint > 0 and iteration > opt.uncertainty_from_iter:
        refl_strength_map = render_pkg.get("refl_strength_map")
        if uncertainty_map is not None and uncertainty_map.numel() > 0 and \
           refl_strength_map is not None and refl_strength_map.numel() > 0:
            loss_hint = l1_loss(uncertainty_map, refl_strength_map)
            tb_dict["loss_hint"] = loss_hint.item()
            loss += opt.lambda_hint * loss_hint
        else:
            tb_dict["loss_hint"] = 0.0
    else:
        tb_dict["loss_hint"] = 0.0
            
    
    if opt.lambda_pseudo_normal > 0 and iteration > opt.pseudo_normal_from_iter and viewpoint_camera.has_normal:
            rendered_normal = render_pkg['rend_normal']
            
            # 1. Foundation Model의 카메라 좌표계 노멀 로드 (shape: H, W, 3)
            gt_normal_cam = viewpoint_camera.normal_map
            
            # --- 아래 블록을 추가하여 좌표계를 보정합니다 ---
            
            # 2. 보정 벡터 정의 (Y축과 Z축을 뒤집는 예시)
            #    (x, y, z) -> (x, -y, -z)
            #    가장 흔한 Y축만 먼저 시도해보세요: correction = torch.tensor([1, -1, 1], ...
            correction_vector = torch.tensor([1, -1, -1], device=gt_normal_cam.device)
            
            # 3. 불러온 Normal에 보정 벡터를 곱해줍니다.
            gt_normal_cam_corrected = gt_normal_cam * correction_vector

            # --- 여기까지 추가 ---

            # 4. (H, W, 3) 형태에서 올바른 높이(H)와 너비(W) 추출
            H, W = gt_normal_cam_corrected.shape[0], gt_normal_cam_corrected.shape[1]

            # 5. 카메라 -> 월드 변환 회전 행렬
            cam_to_world_R = viewpoint_camera.R

            # 6. 행렬 곱셈을 위해 (H, W, 3) -> (H*W, 3) 형태로 변환
            gt_normal_cam_flat = gt_normal_cam_corrected.reshape(-1, 3)
            
            # ... (이하 월드 좌표계 변환 및 손실 계산 코드는 동일) ...
            gt_normal_world_flat = gt_normal_cam_flat @ cam_to_world_R
            gt_normal_world_image = gt_normal_world_flat.reshape(H, W, 3)
            gt_normal = gt_normal_world_image.permute(2, 0, 1)

            mask = (render_pkg['rend_alpha'] > 0).detach()
            pseudo_normal_loss = (1 - (rendered_normal * gt_normal).sum(dim=0, keepdim=True))[mask].mean()
            # print(pseudo_normal_loss)
            
            tb_dict["loss_pseudo_normal"] = pseudo_normal_loss.item()
            loss += opt.lambda_pseudo_normal * pseudo_normal_loss
    else:
        tb_dict["loss_pseudo_normal"] = 0.0
    
    # --- 여기에 새로운 Loss들을 추가합니다 ---

    # Pseudo Diffuse L1 Loss
    if opt.lambda_pseudo_diffuse > 0 and viewpoint_camera.has_pseudo_diffuse:
        rendered_diffuse = render_pkg.get("diffuse_map")
        gt_diffuse = viewpoint_camera.pseudo_diffuse_map
        if rendered_diffuse is not None and gt_diffuse is not None:
            loss_pseudo_diffuse = l1_loss(rendered_diffuse, gt_diffuse)
            loss += opt.lambda_pseudo_diffuse * loss_pseudo_diffuse
            #print("diffuse", opt.lambda_pseudo_diffuse * loss_pseudo_diffuse)
            tb_dict["loss_pseudo_diffuse"] = loss_pseudo_diffuse.item()
    
    # Pseudo Specular L1 Loss
    if opt.lambda_pseudo_specular > 0 and viewpoint_camera.has_pseudo_specular:
        rendered_specular = render_pkg.get("specular_map")
        gt_specular = viewpoint_camera.pseudo_specular_map
        if rendered_specular is not None and gt_specular is not None:
            loss_pseudo_specular = l1_loss(rendered_specular, gt_specular)
            loss += opt.lambda_pseudo_specular * loss_pseudo_specular
            #print("spec", opt.lambda_pseudo_diffuse * loss_pseudo_specular)
            tb_dict["loss_pseudo_specular"] = loss_pseudo_specular.item()
            
    # --- 여기까지 새로운 Loss 추가 ---
    # --- Metallic 맵 기반의 조건부 손실 ---
    # --- Metallic 맵 기반의 조건부 손실 ---
    # <<-- 핵심 수정: iteration > opt.metallic_loss_from_iter 조건을 추가했습니다. -->>
    # --- Metallic 맵 기반의 조건부 손실 ---
    # --- Metallic 맵 기반의 조건부 손실 ---
    if opt.lambda_metallic_supervision > 0 and iteration > opt.metallic_loss_from_iter and viewpoint_camera.has_pseudo_material:
        
        refl_strength_map = render_pkg.get("refl_strength_map")
        diffuse_map = render_pkg.get("diffuse_map")
        roughness_map = render_pkg.get("roughness_map")
        
        # --- ▼▼▼ 이 부분이 핵심 수정 사항입니다 ▼▼▼ ---

        # (H, W) 형태의 단일 채널 material map을 직접 불러옵니다.
        # --- ▼▼▼ 방법 1: SAM 마스크 + 기본값 추가 ▼▼▼ ---

        # 1. (H, W) 형태의 SAM 마스크를 불러옵니다. (bool 또는 0/1 int 타입 가정)
        sam_mask = viewpoint_camera.pseudo_material_map
        
        # 2. 배경에 부여할 기본 반사율을 정의합니다. (하이퍼파라미터)
        base_reflectivity = 0.0
        
        # 3. 마스크를 float 타입으로 변환하고 기본값을 더해줍니다.
        #    객체(1.0)는 1.0을 유지하도록 torch.clamp를 사용합니다.
        single_channel_metallic_map = torch.clamp(
            sam_mask.float()+base_reflectivity, 
            0.0, 
            1.0
        )
        
        # --- ▲▲▲ 여기까지 수정 ▲▲▲ ---
        
        # Loss 계산에 사용하기 위해 채널 차원(1)을 추가하여 (1, H, W) 형태로 만듭니다.
        pseudo_metallic_map = single_channel_metallic_map.unsqueeze(0)
        
        # --- ▲▲▲ 여기까지 수정 ▲▲▲ ---

        if all(m is not None for m in [refl_strength_map, diffuse_map, roughness_map, pseudo_metallic_map]):
            
            # 1단계: refl_strength 직접 감독 (이전과 동일)
            loss_metallic = l1_loss(refl_strength_map, pseudo_metallic_map)
            loss += opt.lambda_metallic_supervision * loss_metallic
            tb_dict["loss_metallic_supervision"] = loss_metallic.item()
            #print("loss_metallic",loss_metallic)

            # 2단계: 조건부 물리 법칙 강제 (Soft Weighting 방식은 그대로 유지)
            soft_metallic_weight = pseudo_metallic_map.detach()
            soft_non_metallic_weight = 1.0 - soft_metallic_weight
            
            # 비금속 영역에 Chroma Loss 적용
            if opt.lambda_purity > 0:
                purity_diffuse = calculate_purity(diffuse_map)
                purity_total = calculate_purity(rendered_image)
                chroma_loss_map = F.relu(purity_total - purity_diffuse)
                masked_chroma_loss = (chroma_loss_map * soft_non_metallic_weight).sum() / (soft_non_metallic_weight.sum() + 1e-8)
                loss += opt.lambda_purity * masked_chroma_loss
                tb_dict["loss_purity_non_metallic"] = masked_chroma_loss.item()

            
            
            # 금속 영역에 Diffuse 억제 손실 적용
            if opt.lambda_diffuse_metal > 0:
                loss_diffuse_metal = (torch.abs(diffuse_map) * soft_metallic_weight).sum() / (soft_metallic_weight.sum() + 1e-8)
                loss += opt.lambda_diffuse_metal * loss_diffuse_metal
                tb_dict["loss_diffuse_metal"] = loss_diffuse_metal.item()
                #print("loss_diffuse_metal",loss_diffuse_metal)

            # 3단계: 조건부 거칠기(Roughness) 손실
            # 금속 영역은 부드럽게 (roughness -> 0)
            if opt.lambda_roughness_metal > 0:
                loss_roughness_metal = (torch.abs(roughness_map) * soft_metallic_weight).sum() / (soft_metallic_weight.sum() + 1e-8)
                loss += opt.lambda_roughness_metal * loss_roughness_metal
                tb_dict["loss_roughness_metal"] = loss_roughness_metal.item()
            
            # 비금속 영역은 거칠게 (roughness -> 1)
            if opt.lambda_roughness_non_metal > 0:
                loss_roughness_non_metal = (torch.abs(1.0 - roughness_map) * soft_non_metallic_weight).sum() / (soft_non_metallic_weight.sum() + 1e-8)
                loss += opt.lambda_roughness_non_metal * loss_roughness_non_metal
                tb_dict["loss_roughness_non_metallic"] = loss_roughness_non_metal.item()
                
                
        if opt.lambda_specular_luminance > 0 and viewpoint_camera.has_pseudo_material:
        
            # 1. 필요한 맵들을 가져옵니다.
            rendered_specular = render_pkg.get("specular_map")
            gt_material_map = viewpoint_camera.pseudo_material_map # (1, H, W)라고 가정
            gt_image = viewpoint_camera.original_image.cuda()

            if rendered_specular is not None:
                
                # 2. 각 맵의 휘도를 계산합니다.
                specular_luminance = rgb_to_luminance(rendered_specular)
                gt_image_luminance = rgb_to_luminance(gt_image)
                
                # 3. GT Material Map을 가중치로 사용하여 L1 손실을 계산합니다.
                #    Material Map 값이 클수록 (금속에 가까울수록) 손실에 더 큰 영향을 줍니다.
                error = torch.abs(specular_luminance - gt_image_luminance)
                weighted_error = error * gt_material_map.detach()
                
                # 가중치가 적용된 평균 손실 계산
                loss_specular_luminance = weighted_error.sum() / (gt_material_map.sum() + 1e-8)
                
                # 4. 전체 손실에 추가합니다.
                loss += opt.lambda_specular_luminance * loss_specular_luminance
                tb_dict["loss_specular_luminance"] = loss_specular_luminance.item()
                #print("loss_specular_luminance",loss_specular_luminance)
        """
        # [추가] 3) High-Frequency Metallic Loss (선명도 강화)
        # "금속이면서(Metallic) + 디테일이 있는(High-Freq)" 부분만 집중 타격
        if opt.lambda_purity > 0:  # lambda_purity 옵션을 같이 쓴다고 가정 (아니면 별도 옵션 추가)
            
            # (1) GT 이미지의 엣지/디테일 추출
            # utils.py에 정의된 create_high_frequency_mask 사용
            gt_hf_mask = create_high_frequency_mask(gt_image).unsqueeze(0) # (1, H, W)
            
            # (2) 타겟 영역 설정 (금속 마스크 AND 고주파 마스크)
            # pseudo_metallic_map은 위에서 이미 정의됨 ((1, H, W))
            # pseudo_metallic_map은 0~1 사이 값이므로 곱하면 AND 연산과 유사한 효과
            target_hf_region = pseudo_metallic_map.detach() * gt_hf_mask
            
            # (3) 해당 영역에 강력한 L1 Loss 적용
            # 렌더링된 이미지와 GT 사이의 차이를 구하되, 타겟 영역에 가중치를 곱함
            hf_weight = 5.0  # 가중치 (실험적으로 조절 가능)
            loss_hf_metal = (torch.abs(rendered_image - gt_image) * target_hf_region).mean() * hf_weight
            
            loss += opt.lambda_metallic_supervision * loss_hf_metal
            tb_dict["loss_hf_metal"] = loss_hf_metal.item()
            """
    
    return loss, tb_dict
