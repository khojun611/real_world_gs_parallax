import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_surfel
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from torchvision.utils import save_image, make_grid
# render.py 상단에 추가
import matplotlib.cm as cm
import torch.nn.functional as F
from utils.image_utils import visualize_depth # 추가
from scene.light import EnvLight

# <<-- H, W 인자를 추가
def apply_colormap(tensor_gray, H, W, cmap='jet', vmin=0.0, vmax=1.0):
    """단일 채널 텐서에 컬러맵을 적용하고 값의 범위를 조절합니다."""
    if tensor_gray.numel() == 0:
        return torch.zeros(3, H, W, device='cuda')

    tensor_gray = tensor_gray.squeeze().cpu().numpy()
    
    tensor_gray = np.clip(tensor_gray, vmin, vmax)
    if vmax - vmin > 0:
        tensor_gray = (tensor_gray - vmin) / (vmax - vmin)
    
    colormap = cm.get_cmap(cmap)
    colored_image = colormap(tensor_gray)[:, :, :3]
    colored_tensor = torch.from_numpy(colored_image).permute(2, 0, 1)
    
    colored_tensor = F.interpolate(colored_tensor.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    
    return colored_tensor.cuda()


def save_debug_grid(render_pkg, viewpoint_cam, save_path, opt):
    """디버깅용 시각화 그리드를 저장하는 함수"""
    error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])
    H, W = error_map.shape[-2:]

        # --- ▼▼▼ 이 부분을 수정하세요 ▼▼▼ ---

    # 1. Uncertainty 맵 시각화 (0~0.1 클리핑 및 jet 컬러맵 적용)
    original_uncertainty_map = render_pkg.get("rgb_uncertainty_map", torch.empty(0, device='cuda'))

    # 'jet' 컬러맵을 사용하고, vmax를 0.1로 설정하여 0~0.1 범위를 클리핑합니다.
    vis_uncertainty = apply_colormap(original_uncertainty_map, H, W, cmap='jet', vmin=0.0, vmax=0.1)

    # --- ▲▲▲ 여기까지 수정 ▲▲▲ ---

    visualization_list = [
        viewpoint_cam.original_image.cuda(),  
        render_pkg["render"],  
        render_pkg.get("base_color_map", torch.zeros_like(error_map)),  
        render_pkg.get("diffuse_map", torch.zeros_like(error_map)),
        render_pkg.get("specular_map", torch.zeros_like(error_map)),
        vis_uncertainty, # <<-- 수정된 시각화 결과 사용
        apply_colormap(render_pkg.get("refl_strength_map", torch.empty(0, device='cuda')), H, W, cmap='jet'),
        apply_colormap(render_pkg.get("roughness_map", torch.empty(0, device='cuda')), H, W, cmap='jet'),
        render_pkg.get("rend_alpha", torch.zeros_like(error_map)).repeat(3, 1, 1),  
        visualize_depth(render_pkg.get("surf_depth", torch.empty(0, device='cuda'))),  
        render_pkg.get("rend_normal", torch.zeros_like(error_map)) * 0.5 + 0.5,  
        render_pkg.get("surf_normal", torch.zeros_like(error_map)) * 0.5 + 0.5,  
        error_map, 
    ]

    grid = make_grid(visualization_list, nrow=5)
    save_image(grid, save_path)
    
def render_all(model_path, views, gaussians, pipeline, background, save_ims, opt):
    render_path = os.path.join(model_path, "all_renders")
    color_path = os.path.join(render_path, 'rgb')
    normal_path = os.path.join(render_path, 'normal')
    diffuse_path = os.path.join(render_path, 'diffuse')
    specular_path = os.path.join(render_path, 'specular')
    uncertainty_path = os.path.join(render_path, 'uncertainty') # .npy 저장용
    vis_uncertainty_path = os.path.join(render_path, 'uncertainty_visualization') # <<-- 1. 시각화 이미지 저장 폴더 경로 추가
    reflection_path = os.path.join(render_path, 'reflection')
    roughness_path  = os.path.join(render_path, 'roughness')
    basecolor_path  = os.path.join(render_path, 'base_color') 
    if save_ims:
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)
        makedirs(diffuse_path, exist_ok=True)
        makedirs(specular_path, exist_ok=True)
        makedirs(uncertainty_path, exist_ok=True)
        makedirs(vis_uncertainty_path, exist_ok=True) # <<-- 2. 시각화 이미지 저장 폴더 생성
        makedirs(reflection_path, exist_ok=True)
        makedirs(roughness_path,  exist_ok=True)
        makedirs(basecolor_path,  exist_ok=True)                # ★ 추가
    total_psnr = 0.0
    for view in tqdm(views, desc="Rendering all views"):
        view.refl_mask = None
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)

        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        
        current_psnr = psnr(render_color, gt_image).mean().double().item()
        total_psnr += current_psnr

        if save_ims:
            torchvision.utils.save_image(render_color, os.path.join(color_path, f'{view.image_name}.png'))
            
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, f'{view.image_name}.png'))
            
            if 'diffuse_map' in rendering:
                diffuse_map = torch.clamp(rendering['diffuse_map'], 0.0, 1.0)
                torchvision.utils.save_image(diffuse_map, os.path.join(diffuse_path, f'{view.image_name}.png'))

            if 'specular_map' in rendering:
                specular_map = torch.clamp(rendering['specular_map'], 0.0, 1.0)
                torchvision.utils.save_image(specular_map, os.path.join(specular_path, f'{view.image_name}.png'))
            
            if 'refl_strength_map' in rendering:
                refl_gray = torch.clamp(rendering['refl_strength_map'], 0.0, 1.0)  # [1,H,W] or [H,W]
                H, W = refl_gray.shape[-2:]
                refl_jet = apply_colormap(refl_gray, H, W, cmap='jet', vmin=0.0, vmax=1.0)  # [3,H,W]
                torchvision.utils.save_image(refl_jet, os.path.join(reflection_path, f'{view.image_name}.png'))

            # ▼ 추가: roughness (0~1) → jet 시각화 PNG 저장
            if 'roughness_map' in rendering:
                rough_gray = torch.clamp(rendering['roughness_map'], 0.0, 1.0)
                H, W = rough_gray.shape[-2:]
                rough_jet = apply_colormap(rough_gray, H, W, cmap='jet', vmin=0.0, vmax=1.0)
                torchvision.utils.save_image(rough_jet, os.path.join(roughness_path, f'{view.image_name}.png'))
                
            bc = None
            if 'base_color_map' in rendering:
                bc = rendering['base_color_map']
            elif 'albedo_map' in rendering:
                bc = rendering['albedo_map']
                
            if bc is not None:
                # 텐서 형태 보정: (H,W,3) -> (3,H,W)
                if bc.ndim == 3 and bc.shape[0] not in (1,3):
                    bc = bc.permute(2,0,1)
                # 단일채널이면 3채널로 복제(보기 편하게)
                if bc.ndim == 2:
                    bc = bc.unsqueeze(0)
                if bc.shape[0] == 1:
                    bc = bc.repeat(3,1,1)
                bc = torch.clamp(bc, 0.0, 1.0)
                torchvision.utils.save_image(bc, os.path.join(basecolor_path, f'{view.image_name}.png'))
            
            # 기존 grid 저장 로직은 주석 처리된 상태로 둡니다.
            # grid_save_path = os.path.join(grid_path, f'{view.image_name}.png')
            # save_debug_grid(rendering, view, grid_save_path, opt)
            
            if "rgb_uncertainty_map" in rendering:
                uncertainty_map = rendering["rgb_uncertainty_map"]
                
                # 1. 원본 uncertainty 데이터를 .npy 파일로 저장
                uncertainty_filename = os.path.join(uncertainty_path, f'{view.image_name}.npy')
                uncertainty_numpy = uncertainty_map.squeeze().cpu().numpy()
                np.save(uncertainty_filename, uncertainty_numpy)

                # --- ▼▼▼ 이 부분이 추가된 내용입니다 ▼▼▼ ---
                
                # 2. 시각화된 uncertainty 이미지를 .png 파일로 저장
                H, W = render_color.shape[-2:] # 이미지 크기 가져오기
                
                # apply_colormap 함수를 사용해 시각화 (jet, 0~0.1 클리핑)
                vis_uncertainty = apply_colormap(uncertainty_map, H, W, cmap='jet', vmin=0.0, vmax=0.1)
                
                # 시각화 이미지 저장 경로 지정
                vis_filename = os.path.join(vis_uncertainty_path, f'{view.image_name}.png')
                
                # 이미지 저장
                torchvision.utils.save_image(vis_uncertainty, vis_filename)
                
                # --- ▲▲▲ 여기까지 추가 ▲▲▲ ---

    if len(views) > 0:
        avg_psnr = total_psnr / len(views)
        print(f"\nAverage PSNR over {len(views)} views: {avg_psnr:.2f} dB")

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op, indirect):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        
        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        if indirect:
            op.indirect = 1
            gaussians.load_mesh_from_ply(dataset.model_path, iteration)

        print("Combining train and test cameras...")
        combined_cameras = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
        
        combined_cameras = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
        all_cameras = combined_cameras                      # ← uid 중복 제거 하지 않음
        all_cameras.sort(key=lambda x: x.image_name)
        
        print(f"Total unique cameras to render: {len(all_cameras)}")
        
        print("Rendering all sorted cameras...")
        render_all(dataset.model_path, all_cameras, gaussians, pipeline, background, save_ims, op)

        env_dict = gaussians.render_env_map()
        if "env1" in env_dict and env_dict["env1"] is not None:
            grid = [env_dict["env1"].permute(2, 0, 1)]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(dataset.model_path, "env1.png"))
        
        if "env2" in env_dict and env_dict["env2"] is not None:
            grid = [env_dict["env2"].permute(2, 0, 1)]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(dataset.model_path, "env2.png"))


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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images, op, True)