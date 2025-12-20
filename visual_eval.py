import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_surfel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from torchvision.utils import save_image, make_grid
import glob, re
import matplotlib.cm as cm
import torch.nn.functional as F
import copy # [추가] 옵션 복사를 위해 필요
from utils.box_utils import save_box_as_ply

# ---- heatmap helper (jet) ----
def apply_colormap(tensor_gray, H, W, cmap='jet', vmin=0.0, vmax=1.0):
    if tensor_gray is None or tensor_gray.numel() == 0:
        return torch.zeros(3, H, W)
    x = tensor_gray.detach().squeeze().float().cpu().numpy()
    x = np.clip(x, vmin, vmax)
    if vmax > vmin:
        x = (x - vmin) / (vmax - vmin)
    colored = cm.get_cmap(cmap)(x)[..., :3]
    colored = torch.from_numpy(colored).permute(2,0,1)
    colored = F.interpolate(colored.unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False).squeeze(0)
    return colored

def find_latest_mesh_iter(model_path):
    ply_list = glob.glob(os.path.join(model_path, "test_*.ply"))
    if not ply_list:
        return None
    iters = []
    for p in ply_list:
        m = re.search(r'test_(\d+)\.ply$', os.path.basename(p))
        if m:
            iters.append(int(m.group(1)))
    return max(iters) if iters else None

def _safe_stem(name):
    base = os.path.basename(str(name))
    stem, _ = os.path.splitext(base)
    return stem.replace(' ', '_')

def _apply_grayscale(tensor_gray, H, W, vmin=0.0, vmax=1.0):
        if tensor_gray is None or tensor_gray.numel() == 0:
            return torch.zeros(3, H, W, device='cuda')
        
        # 1. 값 클램핑 및 정규화
        x = tensor_gray.detach().squeeze().float().clamp(vmin, vmax)
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin) # 0~1 사이 값으로 정규화
            
        # 2. 보간 (Interpolation) - 해상도 맞추기
        x = x.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False).squeeze() # (H, W)
        
        # 3. [핵심] 3채널(RGB)로 복제하여 흑백 이미지 생성
        # (R=G=B=x)
        x_rgb = x.unsqueeze(0).repeat(3, 1, 1) # (3, H, W)
        
        return x_rgb

# [수정] mode 인자를 추가하여 저장 경로를 분리함
def render_set(model_path, views, gaussians, pipeline, background, save_ims, opt, mode="default"):
    import matplotlib.cm as cm
    import torch.nn.functional as F

    def _apply_colormap(tensor_gray, H, W, cmap='jet', vmin=0.0, vmax=1.0):
        if tensor_gray is None or tensor_gray.numel() == 0:
            return torch.zeros(3, H, W, device='cuda')
        x = tensor_gray.detach().squeeze().float().clamp(vmin, vmax)
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        x = x.unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False).squeeze()
        cm_np = cm.get_cmap(cmap)(x.detach().cpu().numpy())[:, :, :3]
        return torch.from_numpy(cm_np).permute(2, 0, 1).to(tensor_gray.device)

    def _safe_stem(name, fallback):
        base = os.path.basename(str(name)) if name is not None else str(fallback)
        stem, _ = os.path.splitext(base)
        return stem if len(stem) > 0 else str(fallback)

    # [수정] 저장 경로에 mode(infinite/parallax)를 포함시킴
    if save_ims:
        render_path   = os.path.join(model_path, "test", f"renders_{mode}") # 폴더 분리
        color_path    = os.path.join(render_path, 'rgb')
        normal_path   = os.path.join(render_path, 'normal')
        base_path     = os.path.join(render_path, 'basecolor')
        diffuse_path  = os.path.join(render_path, 'diffuse')
        specular_path = os.path.join(render_path, 'specular')
        refl_path     = os.path.join(render_path, 'reflection')
        rough_path    = os.path.join(render_path, 'roughness')
        
        os.makedirs(color_path, exist_ok=True)
        os.makedirs(normal_path, exist_ok=True)
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(diffuse_path, exist_ok=True)
        os.makedirs(specular_path, exist_ok=True)
        os.makedirs(refl_path, exist_ok=True)
        os.makedirs(rough_path, exist_ok=True)

    ssims, psnrs, lpipss, render_times = [], [], [], []

    print(f"\n[{mode.upper()}] Rendering Start... (Parallax Correction: {getattr(opt, 'use_parallax_correction', False)})")

    for idx, view in enumerate(tqdm(views, desc=f"Rendering ({mode})")):
        view.refl_mask = None
        t1 = time.time()
        
        # 렌더링 호출 (opt에 따라 Parallax 적용 여부 결정됨)
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        
        render_time = time.time() - t1
        render_times.append(render_time)

        # metrics
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)[None]
        gt           = torch.clamp(view.original_image, 0.0, 1.0).cuda()[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        try:
            lpipss.append(lpips(render_color, gt, net_type='vgg').item())
        except Exception:
            lpipss.append(0.0)

        if save_ims:
            stem = _safe_stem(getattr(view, "image_name", None), f"{idx:05d}")

            # RGB
            torchvision.utils.save_image(render_color, os.path.join(color_path, f'{stem}.png'))
            
            
            # [▼ 추가] Diffuse Map 저장
            if 'diffuse_map' in rendering:
                diffuse = torch.clamp(rendering['diffuse_map'], 0.0, 1.0)
                torchvision.utils.save_image(diffuse, os.path.join(diffuse_path, f'{stem}.png'))

            # [▼ 추가] Base Color Map 저장
            if 'base_color_map' in rendering:
                base_color = torch.clamp(rendering['base_color_map'], 0.0, 1.0)
                torchvision.utils.save_image(base_color, os.path.join(base_path, f'{stem}.png'))
            
            
            # Normal
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, f'{stem}.png'))

            # Specular
            if 'specular_map' in rendering:
                specular = torch.clamp(rendering['specular_map'], 0.0, 1.0)
                torchvision.utils.save_image(specular, os.path.join(specular_path, f'{stem}.png'))

            # Reflection Strength
            # Reflection Strength (Grayscale)
            if 'refl_strength_map' in rendering:
                refl = torch.clamp(rendering['refl_strength_map'], 0.0, 1.0)
                H, W = render_color.shape[-2:]
                
                # [수정] _apply_grayscale 호출
                refl_gray = _apply_grayscale(refl, H, W, vmin=0.0, vmax=1.0)
                
                # 저장
                torchvision.utils.save_image(refl_gray, os.path.join(refl_path, f'{stem}.png'))

            # Roughness (Grayscale)
            if 'roughness_map' in rendering:
                rough = torch.clamp(rendering['roughness_map'], 0.0, 1.0)
                H, W = render_color.shape[-2:]
                
                # [수정] _apply_grayscale 호출
                rough_gray = _apply_grayscale(rough, H, W, vmin=0.0, vmax=1.0)
                
                # 저장
                torchvision.utils.save_image(rough_gray, os.path.join(rough_path, f'{stem}.png'))

            # [▼▼▼ 추가: Parallax Difference Map 저장 (Masked) ▼▼▼]
            if 'parallax_diff_map' in rendering and rendering['parallax_diff_map'] is not None:
                # 1. 차이맵 가져오기 (이미 렌더러에서 마스킹 처리가 되어서 나옵니다!)
                diff_map = rendering['parallax_diff_map'] # (1, H, W)

                # 2. (선택사항) 시각적 효과를 위해 밝기 증폭 (Boosting)
                # 차이가 미세할 수 있으므로 3~5배 정도 곱해서 눈에 잘 띄게 만듭니다.
                vis_diff = torch.clamp(diff_map * 5.0, 0.0, 1.0)
                
                # 3. 컬러맵 적용 (Jet Colormap) -> 빨간색이 큰 차이, 파란색이 작은 차이
                # 흑백으로 보고 싶으면 이 줄을 빼고 그냥 vis_diff를 저장하세요.
                H, W = render_color.shape[-2:]
                vis_diff_color = _apply_colormap(vis_diff, H, W, cmap='jet', vmin=0.0, vmax=1.0)
                
                # 4. 저장 폴더 생성 (loop 밖에서 만드는 게 좋지만 안전하게 여기서 체크)
                diff_path = os.path.join(render_path, 'parallax_diff')
                os.makedirs(diff_path, exist_ok=True)
                
                # 5. 이미지 저장
                torchvision.utils.save_image(vis_diff_color, os.path.join(diff_path, f'{stem}.png'))
            
    # summary
    ssim_v = float(np.mean(ssims)) if ssims else 0.0
    psnr_v = float(np.mean(psnrs)) if psnrs else 0.0
    lpip_v = float(np.mean(lpipss)) if lpipss else 0.0
    fps    = 1.0 / float(np.mean(render_times)) if render_times else 0.0
    
    # [수정] 모드별 결과 출력
    print(f'[{mode.upper()}] PSNR: {psnr_v:.4f}, SSIM: {ssim_v:.4f}, LPIPS: {lpip_v:.4f}, FPS: {fps:.2f}')
    
    # [수정] 파일명도 모드별로 분리
    with open(os.path.join(model_path, f'metric_{mode}.txt'), 'w') as f:
        f.write(f'psnr:{psnr_v}, ssim:{ssim_v}, lpips:{lpip_v}, fps:{fps}')


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op: OptimizationParams, indirect):
    with torch.no_grad():
        print(f"[INFO] model_path={dataset.model_path}")
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # [체크포인트 로드 부분]
        checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{iteration}.pth")
        if os.path.exists(checkpoint_path):
            print(f"[LOAD] Loading optimized parameters (Box, etc.) from: {checkpoint_path}")
            (model_params, first_iter) = torch.load(checkpoint_path)
            
            gaussians.restore(model_params, op) 
            
            # ---------------------------------------------------------------------
            # [▼▼▼ 수정: 박스 정보 디버깅 및 PLY 저장 ▼▼▼]
            # ---------------------------------------------------------------------
            box = gaussians.get_env_box
            
            # 1. 회전 행렬 가져오기 (GaussianModel에 해당 프로퍼티가 있다고 가정)
            rot = None
            if hasattr(gaussians, 'get_env_box_rotation_matrix'):
                rot = gaussians.get_env_box_rotation_matrix
            
            # 2. 로그 출력
            print(f"   >>> Optimized Box Center: {box['center'].cpu().numpy()}")
            print(f"   >>> Optimized Box Min: {box['min'].cpu().numpy()}")
            print(f"   >>> Optimized Box Max: {box['max'].cpu().numpy()}")
            if rot is not None:
                print(f"   >>> Optimized Box Rotation:\n{rot.cpu().numpy()}")
            
            # 3. 렌더링 폴더에 박스 PLY 저장 (시각화용)
            # test 폴더 안에 저장하거나 model_path 루트에 저장
            box_ply_path = os.path.join(dataset.model_path, f"env_box_test_{iteration}.ply")
            save_box_as_ply(
                box_ply_path, 
                box['min'], 
                box['max'], 
                rotation=rot, 
                center=box['center']
            )
            print(f"   >>> Saved env box ply to: {box_ply_path}")
            # ---------------------------------------------------------------------

        else:
            print(f"[WARN] Checkpoint not found at {checkpoint_path}. Using Smart Init Box (from PLY).")
        # ---------------------------------------------------------------------

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Indirect 설정 (메쉬 로드)
        mesh_iter = find_latest_mesh_iter(dataset.model_path)
        if indirect and mesh_iter is not None:
            op.indirect = 1
            ok = gaussians.load_mesh_from_ply(dataset.model_path, mesh_iter)
            if not ok:
                op.indirect = 0
        else:
            op.indirect = 0

        test_cams = scene.getTestCameras()

        # =================================================================
        # 1. Infinite (Parallax Correction OFF) 렌더링
        # =================================================================
        op_infinite = copy.deepcopy(op) # 설정 복사
        op_infinite.use_parallax_correction = False # 강제 OFF
        
        render_set(dataset.model_path, test_cams, gaussians, pipeline, background, save_ims, op_infinite, mode="infinite")

        # =================================================================
        # 2. Parallax Corrected (Parallax Correction ON) 렌더링
        # =================================================================
        op_parallax = copy.deepcopy(op) # 설정 복사
        op_parallax.use_parallax_correction = True # 강제 ON (학습된 박스 사용됨)
        
        render_set(dataset.model_path, test_cams, gaussians, pipeline, background, save_ims, op_parallax, mode="parallax")
        # =================================================================

        # Env Map 저장
        try:
            env_dict = gaussians.render_env_map()
            if "env1" in env_dict and env_dict["env1"] is not None:
                grid = [env_dict["env1"].permute(2, 0, 1)]
                grid = make_grid(grid, nrow=1, padding=10)
                save_image(grid, os.path.join(dataset.model_path, "env1.png"))
            if "env2" in env_dict and env_dict["env2"] is not None:
                grid = [env_dict["env2"].permute(2, 0, 1)]
                grid = make_grid(grid, nrow=1, padding=10)
                save_image(grid, os.path.join(dataset.model_path, "env2.png"))
        except Exception as e:
            print(f"[WARN] render_env_map failed: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    # [중요] op.extract(args)를 사용하여 저장된 파라미터(박스 등)를 불러옵니다.
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images, op.extract(args), True)