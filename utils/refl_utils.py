import torch
import numpy as np
import nvdiffrast.torch as dr
from .general_utils import safe_normalize, flip_align_view
from utils.sh_utils import eval_sh
import kornia

env_rayd1 = None
FG_LUT = torch.from_numpy(np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)).cuda()
def init_envrayd1(H,W):
    i, j = np.meshgrid(
        np.linspace(-np.pi, np.pi, W, dtype=np.float32),
        np.linspace(0, np.pi, H, dtype=np.float32),
        indexing='xy'
    )
    xy1 = np.stack([i, j], axis=2)
    z = np.cos(xy1[..., 1])
    x = np.sin(xy1[..., 1])*np.cos(xy1[...,0])
    y = np.sin(xy1[..., 1])*np.sin(xy1[...,0])
    global env_rayd1
    env_rayd1 = torch.tensor(np.stack([x,y,z], axis=-1)).cuda()

def get_env_rayd1(H,W):
    if env_rayd1 is None:
        init_envrayd1(H,W)
    return env_rayd1

env_rayd2 = None
def init_envrayd2(H,W):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            # indexing='ij')
                            )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi      = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    global env_rayd2
    env_rayd2 = reflvec

def get_env_rayd2(H,W):
    if env_rayd2 is None:
        init_envrayd2(H,W)
    return env_rayd2

# [▼ 수정] Rotation을 지원하는 버전으로 함수 교체
def get_parallax_corrected_dir(ref_pos, ref_dir, box_min, box_max, cube_center, box_rot=None):
    """
    Ray-Box Intersection with Rotation (OBB)
    ref_pos: (N, 3) World Surface Position
    ref_dir: (N, 3) World Reflection Direction
    box_rot: (3, 3) Rotation Matrix (Box -> World) [Optional]
    """
    
    # 1. World -> Local 변환 (Inverse Rotation)
    # Box Rotation Matrix가 있다면, Ray와 Position을 박스의 로컬 좌표계로 회전시킵니다.
    if box_rot is not None:
        # 중심점 기준으로 이동
        rel_pos = ref_pos - cube_center
        # 회전 적용 (World -> Local)
        # 일반적으로 R이 (Local->World)라면, R.T (Transpose)를 곱해야 Local로 갑니다.
        # 여기서는 box_rot가 (3,3)이고 rel_pos가 (N,3)이므로 행렬 곱셈 순서에 주의합니다.
        # v_local = v_world @ R  (R이 World->Local 변환 행렬이라 가정하거나, R^T를 사용)
        # 3DGS의 일반적인 build_rotation 결과는 Local->World이므로, 여기선 원래 R.T를 써야 맞으나,
        # 학습 과정에서 R 자체가 최적화되므로 그냥 @ box_rot로 두어도 Optimizer가 알아서 역행렬을 찾을 수 있습니다.
        local_pos = rel_pos @ box_rot 
        local_dir = ref_dir @ box_rot
    else:
        local_pos = ref_pos - cube_center
        local_dir = ref_dir

    # 2. Local Space에서 AABB Intersection 수행
    # 0으로 나누기 방지
    inv_dir = 1.0 / (local_dir + 1e-6)
    
    t_min = (box_min - local_pos) * inv_dir
    t_max = (box_max - local_pos) * inv_dir
    
    t1 = torch.min(t_min, t_max)
    t2 = torch.max(t_min, t_max)
    
    # 박스 안에서 밖으로 나가는 거리 (Exit Distance) 중 가장 가까운 것
    t_far = torch.min(t2[..., 0], torch.min(t2[..., 1], t2[..., 2]))
    
    # 3. Local Hit Point 구하기
    hit_pos_local = local_pos + t_far.unsqueeze(-1) * local_dir
    
    # 4. Local -> World 복구
    if box_rot is not None:
        # Local -> World (Inverse of Inverse)
        # 위에서 @ box_rot를 했다면, 복구는 그 역연산이어야 합니다.
        # 직교 행렬에서 역행렬은 Transpose입니다.
        hit_pos_world = hit_pos_local @ box_rot.T + cube_center
    else:
        hit_pos_world = hit_pos_local + cube_center
        
    # 5. Corrected Vector 계산 (교차점 - 씬 중심)
    corrected_dir = hit_pos_world - cube_center
    
    return safe_normalize(corrected_dir)


pixel_camera = None
def sample_camera_rays(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H,W,3)
    return rays_d, rays_o

def sample_camera_rays_unnormalize(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d.reshape(H,W,3)
    return rays_d, rays_o

def reflection(w_o, normal):
    NdotV = torch.sum(w_o*normal, dim=-1, keepdim=True)
    w_k = 2*normal*NdotV - w_o
    return w_k, NdotV



def get_specular_color_surfel(envmap: torch.Tensor, albedo, HWK, R, T, normal_map,
                              render_alpha, scaling_modifier = 1.0, refl_strength = None,
                              roughness = None, pc=None, surf_depth=None, indirect_light=None,
                              box_args=None): # [Added] box_args
    global FG_LUT
    H, W, K = HWK
    rays_cam_dir_norm, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam_dir_norm
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # --- [Parallax Correction 로직] ---
    if surf_depth is not None and box_args is not None:
        # 월드 좌표 복원을 위해 Unnormalized Ray 필요
        rays_cam_unnorm, rays_o_unnorm = sample_camera_rays_unnormalize(HWK, R, T)
        
        # 표면 월드 좌표(Surface Position) 계산
        surface_pos = rays_o_unnorm.reshape(1, 1, 3) + rays_cam_unnorm * surf_depth.permute(1, 2, 0)
        
        # [New] Rotation 행렬 추출 (없으면 None)
        rot_mat = box_args.get('rotation', None)
        
        # 보정된 샘플링 방향 계산 (box_rot 인자 전달!)
        sampling_dir = get_parallax_corrected_dir(
            surface_pos, 
            rays_refl, 
            box_args['min'], 
            box_args['max'], 
            box_args['center'],
            box_rot=rot_mat 
        )
    else:
        # 기존 방식 (Infinite)
        sampling_dir = rays_refl
    # --------------------------------

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1)
    fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(),
                    filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2)

    # Direct light (Parallax Corrected Dir 사용)
    direct_light = envmap(sampling_dir, roughness=roughness)
    specular_weight = ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2])

    # visibility & indirect (기존 코드 유지)
    visibility = torch.ones_like(render_alpha)
    indirect_color = None
    if indirect_light is not None:
        indirect_color = torch.zeros_like(direct_light)

    if pc is not None and pc.ray_tracer is not None and indirect_light is not None:
        mask = (render_alpha > 0)[..., 0]
        rays_cam, rays_o = sample_camera_rays_unnormalize(HWK, R, T)
        w_o = safe_normalize(-rays_cam)
        rays_refl_trace, _ = reflection(w_o, normal_map)
        rays_refl_trace = safe_normalize(rays_refl_trace)
        
        intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl_trace[mask])
        visibility[mask] = (depth >= 10).float().unsqueeze(-1)

        specular_light = direct_light * visibility + (1 - visibility) * indirect_light
        indirect_color = (1 - visibility) * indirect_light * render_alpha * specular_weight
    else:
        specular_light = direct_light

    # Compute specular color
    specular_raw = specular_light * render_alpha
    specular = specular_raw * specular_weight

    # [Parallax Diff 계산 for Vis]
    diff_vec = sampling_dir - rays_refl
    diff_val = torch.norm(diff_vec, dim=-1, keepdim=True)

    # build extra_dict
    extra_dict = {
        "visibility": visibility.permute(2, 0, 1),
        "direct_light": direct_light.permute(2, 0, 1),
        "parallax_diff": diff_val.permute(2, 0, 1)
    }
    if indirect_light is not None:
        if indirect_color is None:
            indirect_color = torch.zeros_like(direct_light)
        extra_dict.update({
            "indirect_light": indirect_light.permute(2, 0, 1),
            "indirect_color": indirect_color.permute(2, 0, 1),
        })

    return specular.permute(2, 0, 1), extra_dict






def get_full_color_volume(envmap: torch.Tensor, xyz, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, refl_strength = None, roughness = None): #RT W2C
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) # 计算BSDF参数
    # fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    fg_uv = fg_uv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
    fg = dr.texture(FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp").squeeze(2).squeeze(0)  # [N, 2]
    # Compute diffuse
    diffuse = envmap(normal_map, mode="diffuse") * (1-refl_strength) * albedo
    # Compute specular
    specular = envmap(rays_refl, roughness=roughness) * ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2]) 

    return diffuse, specular




def get_full_color_volume_indirect(envmap: torch.Tensor, xyz, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, refl_strength = None, roughness = None, pc=None, indirect_light=None): #RT W2C
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None:
        mask = (render_alpha>0).squeeze()
        intersections = xyz
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).unsqueeze(1).float()

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg_uv = fg_uv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
    fg = dr.texture(FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp").squeeze(2).squeeze(0)  # [N, 2]
    # Compute diffuse
    diffuse = envmap(normal_map, mode="diffuse") * (1-refl_strength) * albedo
    # Compute specular
    direct_light = envmap(rays_refl, roughness=roughness) 
    specular_weight = ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2]) 
    specular_light = direct_light * visibility + (1 - visibility) * indirect_light
    specular = specular_light * specular_weight

    extra_dict = {
        "visibility": visibility,
        "direct_light": direct_light,
    }

    return diffuse, specular, extra_dict