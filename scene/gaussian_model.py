import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from cubemapencoder import CubemapEncoder
from scene.light import EnvLight
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, init_predefined_omega
from utils.general_utils import strip_symmetric, build_scaling_rotation, safe_normalize, flip_align_view
from utils.refl_utils import sample_camera_rays, get_env_rayd1, get_env_rayd2
import raytracing
import torch.nn.functional as F # <<-- 이 줄을 추가하세요
from utils.general_utils import build_rotation
import torch.nn as nn # 파일 상단에 없다면 추가 필요


def get_env_direction1(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return env_directions


def get_env_direction2(H, W):
    gx, gy = torch.meshgrid(
        torch.linspace(-torch.pi, torch.pi, W, device='cuda'),
        torch.linspace(0, torch.pi, H, device='cuda'),
        indexing='xy'
    )
    env_directions = torch.stack((
        torch.sin(gy)*torch.cos(gx), 
        torch.sin(gy)*torch.sin(gx), 
        torch.cos(gy)
    ), dim=-1)
    return env_directions


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.refl_activation = torch.sigmoid
        self.inverse_refl_activation = inverse_sigmoid

        self.metalness_ativation = torch.sigmoid
        self.inverse_metalness_activation = inverse_sigmoid

        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.asg_param = init_predefined_omega(4, 8)


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._refl_strength = torch.empty(0) 
        self._ori_color = torch.empty(0) 
        self._diffuse_color = torch.empty(0) 
        self._metalness = torch.empty(0) 
        self._roughness = torch.empty(0) 
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._indirect_dc = torch.empty(0)
        self._indirect_rest = torch.empty(0)
        self._indirect_asg = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._uncertainty = torch.empty(0)

        self._normal1 = torch.empty(0)
        self._normal2 = torch.empty(0)

        self.optimizer = None
        self.free_radius = 0    
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.init_refl_value = 0.01
        self.init_roughness_value = 0.1 #[0,1]
        self.init_metalness_value = 0.5 #[0,1]
        self.init_ori_color = 0  
        self.enlarge_scale = 1.5
        self.refl_msk_thr = 0.02
        self.rough_msk_thr = 0.1

        self.env_map = None
        self.env_map_2 = None
        self.env_H, self.env_W = 256, 512
        self.env_directions1 = get_env_direction1(self.env_H, self.env_W)
        self.env_directions2 = get_env_direction2(self.env_H, self.env_W)
        self.ray_tracer = None
        self.setup_functions()
        
        # [▼ 추가] 학습 가능한 환경 박스 파라미터 초기화
        self._env_box_min = torch.tensor([-18.110445, -14.79639,  -15.34217 ], device="cuda")
        self._env_box_max = torch.tensor([18.131962, 14.775374, 15.332282], device="cuda")
        self._env_center = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        
        # [▼ 추가] Regularization을 위한 '초기 상태(Prior)' 저장용 (학습 X)
        self.init_env_box_min = None
        self.init_env_box_max = None
        self.init_env_center = None
        
        self._env_box_quat = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").requires_grad_(True))
        
    # scene/gaussian_model.py 클래스 내부

    def freeze_env_box(self):
        """
        환경 박스(Box) 관련 파라미터의 LR을 0으로 만들어 학습을 중단시킵니다.
        """
        count = 0
        # Optimizer에 등록된 모든 파라미터 그룹을 돕니다.
        for group in self.optimizer.param_groups:
            # 박스 관련 파라미터 이름들 (사용하시는 변수명 모두 포함)
            # 이름이 이 리스트에 포함되면 학습률을 0으로 죽입니다.
            if group["name"] in ["env_box_min", "env_box_max", "env_center", "env_rotation", "env_box_quat"]:
                group["lr"] = 0.0
                count += 1
        
        if count > 0:
            print(f"\n[INFO] ==============================================")
            print(f"[INFO] 🧊 Environment Box parameters have been FROZEN! 🧊")
            print(f"[INFO] (Affected Groups: {count})")
            print(f"[INFO] -> Optimization will now focus only on Texture.")
            print(f"[INFO] ==============================================\n")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._refl_strength, 
            self._metalness, 
            self._roughness, 
            self._ori_color, 
            self._diffuse_color, 
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,  
            self._normal2,  
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self._uncertainty, 
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            # [▼ 추가] 환경 박스 파라미터 저장
            self._env_box_min,
            self._env_box_max,
            self._env_center,
            # [▼▼▼ 추가: EnvMap 상태 저장 ▼▼▼]
            self.env_map.state_dict() if self.env_map is not None else None,
            self.env_map_2.state_dict() if self.env_map_2 is not None else None,
            # [▼▼▼ 여기 아래에 박스 파라미터 4개를 꼭 추가해주세요! ▼▼▼]
            self._env_box_min,
            self._env_box_max,
            self._env_center,
            self._env_rotation
            
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._refl_strength,  
        self._metalness, 
        self._roughness, 
        self._ori_color, 
        self._diffuse_color,
        self._features_dc, 
        self._features_rest,
        self._indirect_dc, 
        self._indirect_rest,
        self._indirect_asg,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._normal1,  
        self._normal2,  
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        self._uncertainty,
        opt_dict, 
        self.spatial_lr_scale,
        *extra_params) = model_args
        
        
        # -----------------------------------------------------------------
        # [추가] 받아낸 extra_params(박스 정보)를 내 변수에 저장
        # -----------------------------------------------------------------
        if len(extra_params) >= 4:
            # -----------------------------------------------------------------
            # [수정] extra_params에서 딕셔너리(opt_dict 등)를 걸러내고 텐서만 추출
            # -----------------------------------------------------------------
            # 1. 텐서인 것만 골라냅니다. (OrderedDict 제외)
            filtered_params = [p for p in extra_params if isinstance(p, torch.Tensor)]
        
            print(f"[INFO] Restoring Environment Box Parameters... (Found {len(extra_params)} extras, {len(filtered_params)} tensors)")

            # [▼▼▼ 수정: 리스트의 '맨 뒤'에서 4개를 가져옵니다 ▼▼▼]
            if len(filtered_params) >= 4:
                # -4, -3, -2, -1 인덱스를 사용합니다.
                self._env_box_min = filtered_params[-4]
                self._env_box_max = filtered_params[-3]
                self._env_center  = filtered_params[-2]
                self._env_rotation = filtered_params[-1]
                
                print("[RESTORE] Env Box Synced (Loaded from the end of list):")
                print(f"   Min: {self._env_box_min.shape}")
                print(f"   Max: {self._env_box_max.shape}")
                print(f"   Rot: {self._env_rotation.shape} (Target: 4)") 
                
            else:
                print("[WARN] No Environment Box parameters found. Using default.")
        
        
        # [▼ 추가] 학습 가능하도록 Parameter로 등록
        self._env_box_min = nn.Parameter(self._env_box_min.requires_grad_(True))
        self._env_box_max = nn.Parameter(self._env_box_max.requires_grad_(True))
        self._env_center = nn.Parameter(self._env_center.requires_grad_(True))
        
        # [▼▼▼ 핵심 수정: 파라미터 등록 및 Init 값 동기화 ▼▼▼]
        # 1. 학습 파라미터로 등록 (Gradient 추적)
        self._env_box_min = nn.Parameter(self._env_box_min.requires_grad_(True))
        self._env_box_max = nn.Parameter(self._env_box_max.requires_grad_(True))
        self._env_center = nn.Parameter(self._env_center.requires_grad_(True))
        
        # 2. Regularization을 위한 기준점(Init값)도 현재 로드된 값으로 업데이트
        # (이렇게 해야 로드된 시점을 기준으로 다시 Regularization이 작동하거나, 고정된 상태로 유지됨)
        self.init_env_box_min = self._env_box_min.detach().clone()
        self.init_env_box_max = self._env_box_max.detach().clone()
        self.init_env_center = self._env_center.detach().clone()
        
        print(f"[RESTORE] Env Box Synced:")
        print(f"   Min: {self.init_env_box_min.cpu().numpy()}")
        print(f"   Max: {self.init_env_box_max.cpu().numpy()}")
        # -------------------------------------------------------------
        
        self._indirect_asg = nn.Parameter(torch.zeros(self._rotation.shape[0], 32, 5, device='cuda').requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        
        # self.optimizer.load_state_dict(opt_dict)

    def set_opacity_lr(self, lr):   
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group['lr'] = lr
    
    
    # [1] 이 함수를 클래스 내부에 추가하세요
    def init_env_box_pca(self):
        """
        포인트 클라우드의 주축(PCA)을 분석하여 Env Box의 회전, 중심, 크기를 자동으로 초기화합니다.
        """
        print("[GaussianModel] Initializing Env Box using PCA...")
        with torch.no_grad():
            # 1. 포인트 데이터 가져오기
            xyz = self.get_xyz.detach()
            if xyz.shape[0] == 0:
                return

            # 2. 중심점 계산 및 이동
            center = xyz.mean(dim=0)
            centered_xyz = xyz - center

            # 3. PCA (공분산 행렬 & 고유값 분해)
            cov = torch.mm(centered_xyz.T, centered_xyz) / (xyz.shape[0] - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            # 4. 회전 행렬 구성 (Eigenvectors가 주축)
            # eigh는 오름차순으로 반환하므로, 컬럼 순서를 확인해야 할 수 있으나 
            # 박스는 축 순서가 바뀌어도 min/max로 커버되므로 그대로 사용합니다.
            rotation_matrix = eigenvectors

            # 좌표계가 뒤집히지 않도록(det=1) 보정 (Right-handed system 유지)
            if torch.det(rotation_matrix) < 0:
                rotation_matrix[:, 2] *= -1

            # 5. Local 좌표계로 변환 (World -> Local)
            # 투영: P_local = P_world @ R (R의 컬럼이 주축일 때)
            local_xyz = torch.matmul(centered_xyz, rotation_matrix)

            # 6. 박스 크기(Extent) 계산 (여유분 1.2배)
            min_xyz = torch.quantile(local_xyz, 0.01, dim=0) * 1.1
            max_xyz = torch.quantile(local_xyz, 0.99, dim=0) * 1.1

            # [수정된 부분 7. 파라미터 생성 및 주입]
            # 기존에 변수가 없으면 nn.Parameter로 새로 정의합니다.
           

            # Center
            if not hasattr(self, "_env_center"):
                self._env_center = nn.Parameter(center.requires_grad_(True))
            else:
                self._env_center.data = center

            # Min
            if not hasattr(self, "_env_box_min"):
                self._env_box_min = nn.Parameter(min_xyz.requires_grad_(True))
            else:
                self._env_box_min.data = min_xyz

            # Max
            if not hasattr(self, "_env_box_max"):
                self._env_box_max = nn.Parameter(max_xyz.requires_grad_(True))
            else:
                self._env_box_max.data = max_xyz
            
            # Rotation (쿼터니언 변환 후 저장)
            q = self._matrix_to_quaternion(rotation_matrix)
            if not hasattr(self, "_env_rotation"):
                self._env_rotation = nn.Parameter(q.requires_grad_(True))
            else:
                self._env_rotation.data = q
            
            print(f" >>> Auto-aligned Center: {center.cpu().numpy()}")
            # print(f" >>> Auto-aligned Extent: \n Min: {min_xyz.cpu().numpy()} \n Max: {max_xyz.cpu().numpy()}")

    # [2] 회전 행렬 -> 쿼터니언 변환 헬퍼 함수 (필요 시 클래스 내부에 추가)
    def _matrix_to_quaternion(self, R):
        """
        3x3 Rotation Matrix를 (w, x, y, z) 쿼터니언으로 변환
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return torch.stack([w, x, y, z])
    
    
    @property
    def get_env_box(self):
        return {
            'min': self._env_box_min,
            'max': self._env_box_max,
            'center': self._env_center
        }
    
    @property
    def get_env_box_rotation_matrix(self):
        """
        저장된 쿼터니언(_env_rotation)을 3x3 회전 행렬로 변환하여 반환
        """
        # 만약 초기화 전이라 변수가 없다면 단위 행렬 반환
        if not hasattr(self, "_env_rotation"):
            return torch.eye(3, device="cuda")
        
        # 1. 쿼터니언 정규화 (학습 중 크기가 변할 수 있으므로 필수)
        q = self._env_rotation
        q = q / (q.norm() + 1e-6)
        
        # 2. Quaternion(w, x, y, z) -> Rotation Matrix 변환 공식
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        R = torch.zeros((3, 3), device=q.device)
        
        R[0, 0] = 1 - 2 * (y**2 + z**2)
        R[0, 1] = 2 * (x * y - z * w)
        R[0, 2] = 2 * (x * z + y * w)
        
        R[1, 0] = 2 * (x * y + z * w)
        R[1, 1] = 1 - 2 * (x**2 + z**2)
        R[1, 2] = 2 * (y * z - x * w)
        
        R[2, 0] = 2 * (x * z - y * w)
        R[2, 1] = 2 * (y * z + x * w)
        R[2, 2] = 1 - 2 * (x**2 + y**2)
        
        return R
    
    @property
    def get_uncertainty(self):
        # softplus는 항상 0 이상의 값을 반환하여 분산 값이 음수가 되는 것을 방지
        return F.softplus(self._uncertainty)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) 
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_refl(self): 
        return self.refl_activation(self._refl_strength)

    @property
    def get_rough(self): 
        return self.roughness_activation(self._roughness)

    @property
    def get_ori_color(self): 
        return self.color_activation(self._ori_color)
    
    @property
    def get_diffuse_color(self): 
        return self.color_activation(self._diffuse_color)
    

    def get_normal(self, scaling_modifier, dir_pp_normalized, return_delta=False): 
        splat2world = self.get_covariance(scaling_modifier)
        normals_raw = splat2world[:,2,:3] 
        normals_raw, positive = flip_align_view(normals_raw, dir_pp_normalized)

        if return_delta:
            delta_normal1 = self._normal1 
            delta_normal2 = self._normal2 
            delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) 
            idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) 
            delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) 
            normals = delta_normal + normals_raw
            normals = safe_normalize(normals) 
            return normals, delta_normal
        else:
            normals = safe_normalize(normals_raw)
            return normals

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_indirect(self):
        indirect_dc = self._indirect_dc
        indirect_rest = self._indirect_rest
        return torch.cat((indirect_dc, indirect_rest), dim=1)
    
    @property
    def get_asg(self):
        return self._indirect_asg
    
    def render_env_map(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map(directions1, mode="pure_env"), 'env2': self.env_map(directions2, mode="pure_env")}
    
    def render_env_map_2(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map_2(directions1, mode="pure_env"), 'env2': self.env_map_2(directions2, mode="pure_env")}

    @property   
    def get_envmap(self): 
        return self.env_map
    
    @property   
    def get_envmap_2(self): 
        return self.env_map_2
    
    @property   
    def get_refl_strength_to_total(self):
        refl = self.get_refl
        return (refl>0.1).sum() / refl.shape[0]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        
        """
        # [▼▼▼ Smart Initialization 구현 (Robust AABB) ▼▼▼]
        print("Computing Robust AABB for EnvMap Parallax Correction...")

        
        # 1. 아웃라이어 제거 (하위 1%, 상위 1% 제거)
        # 노이즈(Floater) 때문에 박스가 무한히 커지는 것을 방지합니다.
        # 1. 데이터 분포 파악 (크기 계산용)
        q_min = torch.quantile(fused_point_cloud, 0.01, dim=0)
        q_max = torch.quantile(fused_point_cloud, 0.99, dim=0)
        
        # [▼ 핵심 수정] 중심점을 무조건 (0, 0, 0)으로 고정
        center = torch.zeros(3, device="cuda")
        
        # 2. 크기 계산
        # 포인트 클라우드의 전체 폭(Width/Height/Depth)을 구합니다.
        size = q_max - q_min
        
        # 3. 마진 적용 (축소)
        # 예: -0.1이면 각 축에서 10%씩 안으로 들어옴 (전체 크기는 20% 감소)
        # 50% 축소를 원하시면 -0.25를 넣으시면 됩니다. (양쪽에서 25%씩 까임)
        margin_ratio = 0.1
        margin = size * margin_ratio
        
        # 4. 원점 기준 대칭 박스 생성
        # (사이즈 / 2) + (음수 마진) = 축소된 반지름(Half Size)
        half_size = (size / 2.0) + margin
        
        safe_min = center - half_size
        safe_max = center + half_size
        
        # [▼ 추가] 안전장치 (뒤집힘 방지)
        min_box_size = 0.2
        current_size = safe_max - safe_min
        
        for i in range(3):
            if current_size[i] < min_box_size:
                # 너무 작아지면 강제로 최소 크기 확보 (중심 0 유지)
                safe_min[i] = -min_box_size / 2.0
                safe_max[i] =  min_box_size / 2.0
        
        # 4. 파라미터에 할당
        self._env_box_min = nn.Parameter(safe_min.requires_grad_(True))
        self._env_box_max = nn.Parameter(safe_max.requires_grad_(True))
        self._env_center = nn.Parameter(center.requires_grad_(True))
        
        # 5. 초기값 저장
        self.init_env_box_min = safe_min.detach().clone()
        self.init_env_box_max = safe_max.detach().clone()
        self.init_env_center = center.detach().clone()
        
        print(f"✅ Smart Init Result (Center Fixed to 0, Shrink {margin_ratio*100}%):")
        print(f"   Min: {safe_min.detach().cpu().numpy()}")
        print(f"   Max: {safe_max.detach().cpu().numpy()}")
        print(f"   Center: {center.detach().cpu().numpy()}")
        """
        
        """
        # 1. 중심점 계산: "중앙값(Median)" 사용 (노이즈 무시)
        # 점들이 아무리 흩뿌려져 있어도, 가장 밀집된 곳(물체)의 중심을 찾습니다.
        center = torch.quantile(fused_point_cloud, 0.5, dim=0)
        
        # 2. 박스 크기 설정: "완벽한 대칭 (Symmetric)"
        # 책상 바닥(-0.2) 같은 가정 없이, 중심에서 모든 방향으로 똑같이 뻗어나갑니다.
        # 3.0m면 가로세로높이 6m짜리 아주 넉넉한 방이 됩니다. (원하시는 대로 조절 가능)
        room_radius = 8.0 
        
        # 3. 박스 생성 (단순 계산)
        # Min = Center - Radius
        # Max = Center + Radius
        safe_min = center - room_radius
        safe_max = center + room_radius
        
        # 4. 파라미터 할당
        self._env_box_min = nn.Parameter(safe_min.requires_grad_(True))
        self._env_box_max = nn.Parameter(safe_max.requires_grad_(True))
        self._env_center = nn.Parameter(center.requires_grad_(True))
        
        # 5. 초기값 저장 (Regularization 및 복원용)
        self.init_env_box_min = safe_min.detach().clone()
        self.init_env_box_max = safe_max.detach().clone()
        self.init_env_center = center.detach().clone()
        
        print(f"✅ Generic Init Result:")
        print(f"   Center (Median): {center.detach().cpu().numpy()}")
        print(f"   Box Radius: {room_radius}m (Symmetric)")
        """
        

        """
        # [▼▼▼ 여기에 원하는 값을 직접 입력하세요 ▼▼▼]
        print("[EnvBox] 🔒 Using Hardcoded Box Size")

        # 예시: X(-3~3), Y(-3~3), Z(-0.2~3) 크기의 박스
        # device="cuda"를 반드시 붙여야 합니다.
        safe_min = torch.tensor([-7.5, -7.5, -7.5], dtype=torch.float, device="cuda")
        safe_max = torch.tensor([ 7.5,  7.5,  7.5], dtype=torch.float, device="cuda")

        # 중심점(Center)은 min/max의 중간으로 다시 계산하는 것이 안전합니다.
        center = (safe_min + safe_max) / 2.0
        # -----------------------------------------------------

        # 4. 파라미터에 할당 (이하는 기존과 동일)
        self._env_box_min = nn.Parameter(safe_min.requires_grad_(True))
        self._env_box_max = nn.Parameter(safe_max.requires_grad_(True))
        self._env_center = nn.Parameter(center.requires_grad_(True))
        
        # 5. 초기값 저장 (Regularization용)
        self.init_env_box_min = safe_min.detach().clone()
        self.init_env_box_max = safe_max.detach().clone()
        self.init_env_center = center.detach().clone()
        
        print(f"✅ Hardcoded Init Result:")
        print(f"   Min: {safe_min.detach().cpu().numpy()}")
        print(f"   Max: {safe_max.detach().cpu().numpy()}")
        print(f"   Center: {center.detach().cpu().numpy()}")
        """
        self._env_box_quat = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").requires_grad_(True))
        
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        sh_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        sh_features[:, :3, 0 ] = fused_color
        sh_features[:, 3:, 1:] = 0.0
        sh_indirect = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        asg_indirect = torch.zeros((fused_color.shape[0], 5, 32)).float().cuda()
        uncertainty = torch.full((fused_point_cloud.shape[0], 1), 1e-4, dtype=torch.float, device="cuda")
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        refl = self.inverse_refl_activation(torch.ones_like(opacities).cuda() * self.init_refl_value)
        refl_strength = refl.cuda()

        metalness = self.inverse_metalness_activation(torch.ones_like(opacities).cuda() * self.init_metalness_value)
        metalness = metalness.cuda()

        roughness = self.inverse_roughness_activation(torch.ones_like(opacities).cuda() * self.init_roughness_value)
        roughness = roughness.cuda()

        def initialize_ori_color(point_cloud, init_color= 0.5, noise_level=0.05):
            base_color = torch.full((point_cloud.shape[0], 3), init_color, dtype=torch.float, device="cuda")
            noise = (torch.rand(point_cloud.shape[0], 3, dtype=torch.float, device="cuda") - 0.5) * noise_level
            ori_color = base_color + noise
            ori_color = torch.clamp(ori_color, 0.0, 1.0)
            return ori_color
        
        ori_color = self.inverse_color_activation(initialize_ori_color(fused_point_cloud))
        diffuse_color = self.inverse_color_activation(initialize_ori_color(fused_point_cloud))  # Initialize diffuse_color similarly

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        
        self.init_env_box_pca()
        
        
        
        self._refl_strength = nn.Parameter(refl_strength.requires_grad_(True))  
        self._ori_color = nn.Parameter(ori_color.requires_grad_(True)) 
        self._diffuse_color = nn.Parameter(diffuse_color.requires_grad_(True))  # Initialize _diffuse_color
        self._roughness = nn.Parameter(roughness.requires_grad_(True)) 
        self._metalness = nn.Parameter(metalness.requires_grad_(True)) 
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._uncertainty = nn.Parameter(uncertainty.requires_grad_(True)) # <<-- 파라미터로 등록
        self._features_dc = nn.Parameter(sh_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(sh_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_dc = nn.Parameter(sh_indirect[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(sh_indirect[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_asg = nn.Parameter(asg_indirect.transpose(1, 2).contiguous().requires_grad_(True))
        
        normals1 = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
        normals2 = np.copy(normals1)
        self._normal1 = nn.Parameter(torch.from_numpy(normals1).to(self._xyz.device).requires_grad_(True))
        self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(True))

        self.env_map = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
        self.env_map_2 = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args, projection_head=None):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        
        
        # [▼ 추가] Parameter로 확실히 등록 (Restore를 안 거쳤을 경우 대비)
        if not isinstance(self._env_box_min, nn.Parameter):
            self._env_box_min = nn.Parameter(self._env_box_min.requires_grad_(True))
            self._env_box_max = nn.Parameter(self._env_box_max.requires_grad_(True))
            self._env_center = nn.Parameter(self._env_center.requires_grad_(True))
            
        # [▼ 추가] 파라미터 안전 장치
        if not isinstance(self._env_box_quat, nn.Parameter):
             self._env_box_quat = nn.Parameter(self._env_box_quat.requires_grad_(True))

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},     
            {'params': self.env_map_2.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env2"}     
        ]

        
        
        
        
        self._normal1.requires_grad_(requires_grad=False)
        self._normal2.requires_grad_(requires_grad=False)
        l.extend([
            {'params': [self._refl_strength], 'lr': training_args.refl_strength_lr, "name": "refl_strength"},  
            {'params': [self._ori_color], 'lr': training_args.ori_color_lr, "name": "ori_color"},  
            {'params': [self._diffuse_color], 'lr': training_args.ori_color_lr, "name": "diffuse_color"},  
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},  
            {'params': [self._metalness], 'lr': training_args.metalness_lr, "name": "metalness"}, 
            {'params': [self._uncertainty], 'lr': training_args.uncertainty_lr, "name": "uncertainty"}, # <<-- 옵티마이저에 추가 
            {'params': [self._normal1], 'lr': training_args.normal_lr, "name": "normal1"},
            {'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"},
            {'params': [self._indirect_dc], 'lr': training_args.indirect_lr, "name": "ind_dc"},
            {'params': [self._indirect_rest], 'lr': training_args.indirect_lr / 20.0, "name": "ind_rest"},
            {'params': [self._indirect_asg], 'lr': training_args.asg_lr, "name": "ind_asg"},
        ])
        # <<--- 바로 이 부분을 추가해야 합니다 ---<<
        if projection_head is not None:
            l.append({'params': projection_head.parameters(), 'lr': training_args.contrastive_lr, "name": "projection_head"})

        # [▼▼▼ Box Parameter 학습 여부 토글 (Training Toggle) ▼▼▼]
        if getattr(training_args, "train_env_box", False): # 옵션이 True인지 확인
            print(f"[INFO] Environment Box Training Enabled! (LR: {training_args.env_box_lr})")
            
            # 1. Min/Max/Center 추가
            if hasattr(self, "_env_box_min"):
                l.append({'params': [self._env_box_min], 'lr': training_args.env_box_lr, "name": "env_box_min"})
                l.append({'params': [self._env_box_max], 'lr': training_args.env_box_lr, "name": "env_box_max"})
                l.append({'params': [self._env_center],  'lr': training_args.env_box_lr, "name": "env_center"})
            
            # 2. Rotation 추가 (이게 있어야 회전도 학습됨!)
            if hasattr(self, "_env_rotation"):
                l.append({'params': [self._env_rotation], 'lr': training_args.env_box_lr, "name": "env_rotation"})
        else:
             print("[INFO] Environment Box Training DISABLED.")
        
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    
    def get_box_reg_loss(self):
        """
        박스가 초기값(Smart Init)에서 너무 멀어지거나 무한대로 커지는 것을 방지하는 MSE Loss
        """
        if self.init_env_box_min is None:
            return torch.tensor(0.0, device="cuda")
            
        # 초기값과의 거리(MSE) 계산
        loss_min = torch.mean((self._env_box_min - self.init_env_box_min) ** 2)
        loss_max = torch.mean((self._env_box_max - self.init_env_box_max) ** 2)
        loss_center = torch.mean((self._env_center - self.init_env_center) ** 2)
        
        return loss_min + loss_max + loss_center
    
    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz','nx2', 'ny2', 'nz2']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._indirect_dc.shape[1]*self._indirect_dc.shape[2]):
            l.append('ind_dc_{}'.format(i))
        for i in range(self._indirect_rest.shape[1]*self._indirect_rest.shape[2]):
            l.append('ind_rest_{}'.format(i))
        for i in range(self._indirect_asg.shape[1]*self._indirect_asg.shape[2]):
            l.append('ind_asg_{}'.format(i))
        l.append('opacity')
        l.append('refl_strength') 
        l.append('metalness') 
        l.append('roughness')
        l.append('uncertainty') 
        for i in range(self._ori_color.shape[1]):
            l.append('ori_color_{}'.format(i))
        for i in range(self._diffuse_color.shape[1]):  # Add diffuse_color attributes
            l.append('diffuse_color_{}'.format(i))


        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_dc = self._indirect_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_rest = self._indirect_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_asg = self._indirect_asg.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        refl_strength = self._refl_strength.detach().cpu().numpy()    
        metalness = self._metalness.detach().cpu().numpy()    
        roughness = self._roughness.detach().cpu().numpy()
        uncertainty = self._uncertainty.detach().cpu().numpy()    
        ori_color = self._ori_color.detach().cpu().numpy()    
        diffuse_color = self._diffuse_color.detach().cpu().numpy()  
        
        normals1 = self._normal1.detach().cpu().numpy()
        normals2 = self._normal2.detach().cpu().numpy() 

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals1, normals2, f_dc, f_rest, ind_dc, ind_rest, ind_asg, opacities, refl_strength, metalness, roughness,uncertainty, ori_color, diffuse_color, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        if self.env_map is not None:
            save_path = path.replace('.ply', '1.map')
            torch.save(self.env_map.state_dict(), save_path)

        if self.env_map_2 is not None:
            save_path = path.replace('.ply', '2.map')
            torch.save(self.env_map_2.state_dict(), save_path)
                

    def reset_opacity0(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1(self, exclusive_msk = None):
        RESET_V = 0.9
        opacity_old = self.get_opacity
        o_msk = (opacity_old > RESET_V).flatten()
        if exclusive_msk is not None:
            o_msk = torch.logical_or(o_msk, exclusive_msk)
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([RESET_V]).cuda())
        opacities_new[o_msk] = self._opacity[o_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1_strategy2(self):
        RESET_B = 1.5
        opacity_old = self.get_opacity
        opacities_new = inverse_sigmoid((opacity_old*RESET_B).clamp(0,0.99))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]


    def reset_refl(self, exclusive_msk = None):
        refl_new = inverse_sigmoid(torch.max(self.get_refl, torch.ones_like(self.get_refl)*self.init_refl_value))
        if exclusive_msk is not None:
            refl_new[exclusive_msk] = self._refl_strength[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(refl_new, "refl_strength")
        if "refl_strength" not in optimizable_tensors: return
        self._refl_strength = optimizable_tensors["refl_strength"]


    def dist_rot(self): 
        REFL_MSK_THR = self.refl_msk_thr
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        rot = self.get_rotation.clone()
        dist_rot = self.rotation_activation(rot + torch.randn_like(rot)*0.08)
        dist_rot[refl_msk] = rot[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_rot, "rotation")
        if "rotation" not in optimizable_tensors: return
        self._rotation = optimizable_tensors["rotation"]

    def dist_albedo(self, exclusive_msk = None):
        REFL_MSK_THR = self.refl_msk_thr
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._ori_color.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "ori_color")
        if "ori_color" not in optimizable_tensors: return
        self._ori_color = optimizable_tensors["ori_color"]

    def dist_color(self, exclusive_msk = None):
        REFL_MSK_THR = self.refl_msk_thr
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._features_dc.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "f_dc")
        if "f_dc" not in optimizable_tensors: return
        self._features_dc = optimizable_tensors["f_dc"]

    def enlarge_refl_scales(self, ret_raw=True, ENLARGE_SCALE=1.5, REFL_MSK_THR=0.02, ROUGH_MSK_THR=0.1, exclusive_msk=None):
        ENLARGE_SCALE = self.enlarge_scale
        REFL_MSK_THR = self.refl_msk_thr
        ROUGH_MSK_THR = self.rough_msk_thr

        refl_msk = self.get_refl.flatten() < REFL_MSK_THR
        rough_msk = self.get_rough.flatten() > ROUGH_MSK_THR
        combined_msk = torch.logical_or(refl_msk, rough_msk)
        if exclusive_msk is not None:
            combined_msk = torch.logical_or(combined_msk, exclusive_msk) 
        scales = self.get_scaling
        rmin_axis = (torch.ones_like(scales) * ENLARGE_SCALE)
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scales * rmin_axis)
            scale_new[combined_msk] = self._scaling[combined_msk]
        else:
            scale_new = scales * rmin_axis
            scale_new[combined_msk] = scales[combined_msk]   
        return scale_new

    def reset_scale(self, exclusive_msk = None):
        scale_new = self.enlarge_refl_scales(ret_raw=True, exclusive_msk=exclusive_msk)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        if "scaling" not in optimizable_tensors: return
        self._scaling = optimizable_tensors["scaling"]


    def reset_features(self, reset_value_dc=0.0, reset_value_rest=0.0):
        # 重置 features_dc
        features_dc_new = torch.full_like(self._features_dc, reset_value_dc, dtype=torch.float, device="cuda")
        # 重置 features_rest
        features_rest_new = torch.full_like(self._features_rest, reset_value_rest, dtype=torch.float, device="cuda")

        # 将新的features_dc和features_rest替换到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(features_dc_new, "f_dc")
        optimizable_tensors.update(self.replace_tensor_to_optimizer(features_rest_new, "f_rest"))
        # 更新active_sh_degree
        self.active_sh_degree = 0

        # 更新类中的属性
        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]


    def reset_ori_color(self, reset_value=0.5, noise_level=0.05):
        base_color = torch.full_like(self._ori_color, reset_value, dtype=torch.float, device="cuda")
        noise = (torch.rand_like(base_color, dtype=torch.float, device="cuda") - 0.5) * noise_level
        ori_color_new = base_color + noise
        ori_color_new = torch.clamp(ori_color_new, 0.0, 1.0)
        
        # 将重置后的 ori_color 更新到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_color_activation(ori_color_new), "ori_color")
        if "ori_color" in optimizable_tensors:
            self._ori_color = optimizable_tensors["ori_color"]

    def reset_refl_strength(self, reset_value=0.01):
        refl_strength_new = torch.full_like(self._refl_strength, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_refl_activation(refl_strength_new), "refl_strength")
        if "refl_strength" in optimizable_tensors:
            self._refl_strength = optimizable_tensors["refl_strength"]
    
    def reset_roughness(self, reset_value=0.1):
        roughness_new = torch.full_like(self._roughness, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_roughness_activation(roughness_new), "roughness")
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]


    def load_ply(self, path, relight=False, args=None):
        plydata = PlyData.read(path)
        

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # # 
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        refl_strength = np.asarray(plydata.elements[0]["refl_strength"])[..., np.newaxis] # #

        ori_color = np.stack((np.asarray(plydata.elements[0]['ori_color_0']),
                              np.asarray(plydata.elements[0]['ori_color_1']),
                              np.asarray(plydata.elements[0]['ori_color_2'])),  axis=1)
        diffuse_color = np.stack((np.asarray(plydata.elements[0]['diffuse_color_0']),
                                np.asarray(plydata.elements[0]['diffuse_color_1']),
                                np.asarray(plydata.elements[0]['diffuse_color_2'])),  axis=1)
        
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis] # #
        metalness = np.asarray(plydata.elements[0]["metalness"])[..., np.newaxis] # #

        normal1 = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)
        normal2 = np.stack((np.asarray(plydata.elements[0]["nx2"]),
                        np.asarray(plydata.elements[0]["ny2"]),
                        np.asarray(plydata.elements[0]["nz2"])),  axis=1)


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        self.active_sh_degree = self.max_sh_degree
        
        indirect_dc = np.zeros((xyz.shape[0], 3, 1))
        indirect_dc[:, 0, 0] = np.asarray(plydata.elements[0]["ind_dc_0"])
        indirect_dc[:, 1, 0] = np.asarray(plydata.elements[0]["ind_dc_1"])
        indirect_dc[:, 2, 0] = np.asarray(plydata.elements[0]["ind_dc_2"])

        extra_ind_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_rest_")]
        extra_ind_names = sorted(extra_ind_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_ind_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        indirect_extra = np.zeros((xyz.shape[0], len(extra_ind_names)))
        for idx, attr_name in enumerate(extra_ind_names):
            indirect_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        indirect_extra = indirect_extra.reshape((indirect_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        extra_asg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_asg_")]
        extra_asg_names = sorted(extra_asg_names, key = lambda x: int(x.split('_')[-1]))
        indirect_asg = np.zeros((xyz.shape[0], len(extra_asg_names)))
        for idx, attr_name in enumerate(extra_asg_names):
            indirect_asg[:, idx] = np.asarray(plydata.elements[0][attr_name])
        indirect_asg = indirect_asg.reshape((indirect_asg.shape[0], 5, -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # #
        if not relight:
            map_path1 = path.replace('.ply', '1.map')
            map_path2 = path.replace('.ply', '2.map')
            if os.path.exists(map_path1)  and os.path.exists(map_path2):
                # self.env_map = CubemapEncoder(output_dim=3, resolution=128).cuda()
                self.env_map = EnvLight(path=None, device='cuda',  max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map.load_state_dict(torch.load(map_path1))
                self.env_map.build_mips()
                self.env_map_2 = EnvLight(path=None, device='cuda',  max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map_2.load_state_dict(torch.load(map_path2))
                self.env_map_2.build_mips()
        else:
            map_path = path.replace('.ply', '.hdr')
            self.env_map = EnvLight(path=map_path, device='cuda', trainable=True).cuda()


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        self._refl_strength = nn.Parameter(torch.tensor(refl_strength, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._metalness = nn.Parameter(torch.tensor(metalness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._ori_color = nn.Parameter(torch.tensor(ori_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._diffuse_color = nn.Parameter(torch.tensor(diffuse_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #

        self._normal1 = nn.Parameter(torch.tensor(normal1, dtype=torch.float, device="cuda").requires_grad_(True))       # #
        self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True))       # #

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._indirect_dc = nn.Parameter(torch.tensor(indirect_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(torch.tensor(indirect_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_asg = nn.Parameter(torch.tensor(indirect_asg, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        
        props = [p.name for p in plydata.elements[0].properties]
        if "uncertainty" not in props:
            print("No uncertainty found in PLY, initializing to default.")
            num_points = xyz.shape[0]  # ← self._xyz가 아니라 xyz 길이를 사용
            uncertainty = torch.full((num_points, 1), 1e-4, dtype=torch.float, device="cuda")
            self._uncertainty = nn.Parameter(uncertainty.requires_grad_(True))
        else:
            uncertainty_np = np.asarray(plydata.elements[0]["uncertainty"])[..., np.newaxis]
            self._uncertainty = nn.Parameter(
                torch.tensor(uncertainty_np, dtype=torch.float, device="cuda").requires_grad_(True)
            )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["mlp", "env", "env2", "projection_head", "env_box_min", "env_box_max", "env_center", "env_box_quat"]: 
                continue
            # 2. [강력 추천] 길이로 확인 (Shape Mismatch 방지)
            # 파라미터 길이가 마스크(점 개수)와 다르면 점 데이터가 아니므로 무조건 건너뜁니다.
            # 이 코드가 있으면 tensor [4] 같은 게 들어와도 안전하게 넘어갑니다.
            if len(group["params"][0]) != len(mask):
                continue
            # -------------------------------------------------------
            
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._uncertainty = optimizable_tensors["uncertainty"]
        self._refl_strength = optimizable_tensors['refl_strength']    # #
        self._ori_color = optimizable_tensors['ori_color']    # #
        self._diffuse_color = optimizable_tensors['diffuse_color']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._metalness = optimizable_tensors['metalness']    # #
        self._normal1 = optimizable_tensors["normal1"]        # #
        self._normal2 = optimizable_tensors["normal2"]        # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._indirect_asg = optimizable_tensors["ind_asg"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["mlp", "env", "env2", "projection_head", "env_box_min", "env_box_max", "env_center", "env_box_quat"]: 
                continue
            
            if 'env_' in group["name"] or group["name"] not in tensors_dict:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_refl_strength, new_metalness, new_roughness, new_ori_color, new_diffuse_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacities, new_scaling, new_rotation, new_normal1, new_normal2, new_uncertainty):
        d = {"xyz": new_xyz,
             
        "refl_strength": new_refl_strength,    # #
        "metalness": new_metalness,    # #
        "roughness": new_roughness,    # #
        "ori_color": new_ori_color,    # #
        "diffuse_color": new_diffuse_color,    # #
        "normal1" : new_normal1,       # #
        "normal2" : new_normal2,       # #

        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        
        "ind_dc": new_indirect_dc,
        "ind_rest": new_indirect_rest,
        "ind_asg": new_indirect_asg,
        
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "uncertainty": new_uncertainty}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._uncertainty = optimizable_tensors["uncertainty"]
        self._refl_strength = optimizable_tensors['refl_strength']    # #
        self._metalness = optimizable_tensors['metalness']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._ori_color = optimizable_tensors['ori_color']    # #
        self._diffuse_color = optimizable_tensors['diffuse_color']    # #
        self._normal1 = optimizable_tensors["normal1"]        # #
        self._normal2 = optimizable_tensors["normal2"]        # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._indirect_asg = optimizable_tensors["ind_asg"]
        
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._uncertainty = optimizable_tensors["uncertainty"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_refl_strength = self._refl_strength[selected_pts_mask].repeat(N,1)   # #
        new_ori_color = self._ori_color[selected_pts_mask].repeat(N,1)   # #
        new_diffuse_color = self._diffuse_color[selected_pts_mask].repeat(N,1)   # #
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)   # #
        new_metalness = self._metalness[selected_pts_mask].repeat(N,1)   # #
        new_normal1 = self._normal1[selected_pts_mask].repeat(N,1)        # #
        new_normal2 = self._normal2[selected_pts_mask].repeat(N,1)       # #
        new_uncertainty = self._uncertainty[selected_pts_mask].repeat(N,1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        
        new_indirect_dc = self._indirect_dc[selected_pts_mask].repeat(N,1,1)
        new_indirect_rest = self._indirect_rest[selected_pts_mask].repeat(N,1,1)
        new_indirect_asg = self._indirect_asg[selected_pts_mask].repeat(N,1,1)
        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_refl_strength, new_metalness, new_roughness, new_ori_color, new_diffuse_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacity, new_scaling, new_rotation, new_normal1, new_normal2, new_uncertainty)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        new_refl_strength = self._refl_strength[selected_pts_mask]   # #
        new_metalness = self._metalness[selected_pts_mask]   # #
        new_roughness = self._roughness[selected_pts_mask]   # #
        new_ori_color = self._ori_color[selected_pts_mask]   # #
        new_diffuse_color = self._diffuse_color[selected_pts_mask]   # #
        new_normal1 = self._normal1[selected_pts_mask]       # #
        new_normal2 = self._normal2[selected_pts_mask]       # #
        new_uncertainty = self._uncertainty[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        
        new_indirect_dc = self._indirect_dc[selected_pts_mask]
        new_indirect_rest = self._indirect_rest[selected_pts_mask]
        new_indirect_asg = self._indirect_asg[selected_pts_mask]
        
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_refl_strength, new_metalness, new_roughness, new_ori_color, new_diffuse_color, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacities, new_scaling, new_rotation, new_normal1, new_normal2, new_uncertainty)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)  # #
        self.denom[update_filter] += 1

    # #
    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state
        
    def update_mesh(self, mesh):
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        if vertices.size == 0 or faces.size == 0:
            print("[ERROR] Cannot build RayTracer: empty vertices/faces")
            self.ray_tracer = None
            return
        self.ray_tracer = raytracing.RayTracer(vertices, faces)

    def load_mesh_from_ply(self, model_path, iteration):
        import open3d as o3d
        import os

        ply_path = os.path.join(model_path, f'test_{iteration:06d}.ply')
        if not os.path.exists(ply_path):
            print(f"[ERROR] Mesh file not found: {ply_path}")
            self.ray_tracer = None
            return False
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if mesh.is_empty() or len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
            print(f"[ERROR] Empty mesh loaded: {ply_path}")
            self.ray_tracer = None
            return False
        self.update_mesh(mesh)
        return True
    