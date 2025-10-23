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

# eval.py 파일 내

# eval.py 파일 내

# <<-- H, W 인자를 추가
def apply_colormap(tensor_gray, H, W, cmap='jet', vmin=0.0, vmax=1.0):
    """단일 채널 텐서에 컬러맵을 적용하고 값의 범위를 조절합니다."""
    if tensor_gray.numel() == 0:
        return torch.zeros(3, H, W, device='cuda')

    tensor_gray = tensor_gray.squeeze().cpu().numpy()
    
    # <<-- 1. 지정된 범위(vmin, vmax)로 값을 제한(clip)합니다.
    tensor_gray = np.clip(tensor_gray, vmin, vmax)
    
    # <<-- 2. 최솟값/최댓값 대신 vmin/vmax를 기준으로 정규화합니다.
    if vmax - vmin > 0:
        tensor_gray = (tensor_gray - vmin) / (vmax - vmin)
    
    colormap = cm.get_cmap(cmap)
    colored_image = colormap(tensor_gray)[:, :, :3]
    colored_tensor = torch.from_numpy(colored_image).permute(2, 0, 1)
    
    colored_tensor = F.interpolate(colored_tensor.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    
    return colored_tensor.cuda()

# eval.py 파일 내


def save_debug_grid(render_pkg, viewpoint_cam, save_path):
    """디버깅용 시각화 그리드를 저장하는 함수"""
    error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])
    # <<-- 여기서 이미지의 높이(H)와 너비(W)를 가져옵니다.
    H, W = error_map.shape[-2:]
    
    visualization_list = [
        viewpoint_cam.original_image.cuda(),  
        render_pkg["render"],  
        render_pkg.get("base_color_map", torch.zeros_like(error_map)),  
        render_pkg.get("diffuse_map", torch.zeros_like(error_map)),
        render_pkg.get("specular_map", torch.zeros_like(error_map)),
        # <<-- apply_colormap 호출 시 H, W를 전달합니다.
        
        # train.py와 동일하게 vmax=0.5를 지정
        apply_colormap(render_pkg.get("rgb_uncertainty_map", ...), H, W, cmap='jet', vmax=0.5),
        apply_colormap(render_pkg.get("refl_strength_map", torch.empty(0, device='cuda')), H, W, cmap='jet'),
        apply_colormap(render_pkg.get("roughness_map", torch.empty(0, device='cuda')), H, W, cmap='jet'),
        render_pkg.get("rend_alpha", torch.zeros_like(error_map)).repeat(3, 1, 1),  
        visualize_depth(render_pkg.get("surf_depth", torch.empty(0, device='cuda'))),  
        render_pkg.get("rend_normal", torch.zeros_like(error_map)) * 0.5 + 0.5,  
        render_pkg.get("surf_normal", torch.zeros_like(error_map)) * 0.5 + 0.5,  
        error_map, 
    ]
    #print(render_pkg.get("rgb_uncertainty_map", torch.empty(0, device='cuda')))
    #print(render_pkg.get("roughness_map", torch.empty(0, device='cuda')))
    grid = make_grid(visualization_list, nrow=5)
    save_image(grid, save_path)
    
# render_set 함수를 render_all로 이름을 바꾸고 수정합니다.
def render_all(model_path, views, gaussians, pipeline, background, save_ims, opt):
    render_path = os.path.join(model_path, "all_renders")
    color_path = os.path.join(render_path, 'rgb')
    normal_path = os.path.join(render_path, 'normal')
    grid_path = os.path.join(render_path, 'grid')

    if save_ims:
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)
        makedirs(grid_path, exist_ok=True)

    for view in tqdm(views, desc="Rendering all views"):
        # print(f"Rendering UID: {view.uid}, Image Name: {view.image_name}") # 디버깅용 print는 그대로 두셔도 됩니다.
        total_psnr = 0.0
        view.refl_mask = None
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        # --- PSNR 계산 로직 추가 ---
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        
        current_psnr = psnr(render_color, gt_image).mean().double().item()
        total_psnr += current_psnr
        # ---------------------------

        if save_ims:
            # 최종 RGB 이미지 저장
            render_color = torch.clamp(rendering["render"], 0.0, 1.0)
            ### 핵심 수정: 파일명을 view.image_name으로 변경 ###
            torchvision.utils.save_image(render_color, os.path.join(color_path, f'{view.image_name}.png'))
            
            # 최종 Normal 맵 저장
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                ### 핵심 수정: 파일명을 view.image_name으로 변경 ###
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, f'{view.image_name}.png'))
            
            # 요소별 디버깅 그리드 저장
            grid_save_path = os.path.join(grid_path, f'{view.image_name}.png')
            save_debug_grid(rendering, view, grid_save_path)
        
        # 모든 뷰에 대한 렌더링이 끝난 후, 평균 PSNR 계산 및 출력
        if len(views) > 0:
            avg_psnr = total_psnr / len(views)
            print(f"\nAverage PSNR over {len(views)} views: {avg_psnr:.2f} dB")

def render_set_train(model_path, views, gaussians, pipeline, background, save_ims, opt):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "train", "renders")
        color_path = os.path.join(render_path, 'rgb')
        gt_path = os.path.join(render_path, 'gt')
        normal_path = os.path.join(render_path, 'normal')
        makedirs(color_path, exist_ok=True)
        makedirs(gt_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
 
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, :3, :, :]

        if save_ims:
            # Save the rendered color image
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}.png'.format(idx)))
            # Save the normal map if available
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
            

            

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op, indirect):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        # 1. 불러올 체크포인트(.pth)의 반복 횟수(iteration)를 결정합니다.
        if iteration == -1:
            try:
                chkpnt_files = [f for f in os.listdir(dataset.model_path) if f.startswith("chkpnt") and f.endswith(".pth")]
                if not chkpnt_files:
                    print(f"[ERROR] No checkpoint (.pth) files found in '{dataset.model_path}'.")
                    return
                iters = [int(f.replace("chkpnt", "").split('.')[0].replace("_uc", "")) for f in chkpnt_files]
                loaded_iter = max(iters)
            except Exception as e:
                print(f"[ERROR] Could not find or parse checkpoint files in '{dataset.model_path}': {e}")
                return
        else:
            loaded_iter = iteration

        # 2. Scene 객체를 먼저 초기화하여 .ply와 .map(환경맵)을 불러옵니다.
        #    이 과정을 통해 gaussians.env_map이 정상적으로 설정됩니다.
        print(f"[INFO] Initializing scene with data from iteration {loaded_iter} to load envmap...")
        scene = Scene(dataset, gaussians, load_iteration=loaded_iter, shuffle=False)

        # 3. .pth 체크포인트 파일을 불러옵니다.
        checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{loaded_iter}.pth")
        uc_checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{loaded_iter}_uc.pth")
        if os.path.exists(uc_checkpoint_path):
            checkpoint_path = uc_checkpoint_path
        
        print(f"[INFO] Loading full model state from checkpoint: {checkpoint_path}")
        (model_params, _) = torch.load(checkpoint_path)
        
        # 4. .pth 파일의 파라미터로 모델을 최종 복원합니다.
        #    이 시점에는 gaussians.env_map이 이미 존재하므로 오류가 발생하지 않습니다.
        gaussians.restore(model_params, op)

        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        
        if indirect:
            op.indirect = 1
            try:
                mesh_files = [f for f in os.listdir(dataset.model_path) if f.startswith("test_") and f.endswith(".ply")]
                if mesh_files:
                    mesh_iters = [int(f.replace("test_", "").replace(".ply", "")) for f in mesh_files]
                    mesh_iteration = max(mesh_iters)
                    gaussians.load_mesh_from_ply(dataset.model_path, mesh_iteration)
            except Exception:
                print("[WARNING] Could not load mesh file. Continuing without it.")

        # (이하 카메라 리스트 생성 및 render_all 호출 코드는 기존과 동일)
        print("Combining train and test cameras...")
        combined_cameras = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
        
        unique_cameras = {cam.uid: cam for cam in combined_cameras}
        all_cameras = list(unique_cameras.values())
        all_cameras.sort(key=lambda x: x.image_name)
        
        print(f"Total unique cameras to render: {len(all_cameras)}")
        
        print("Rendering all sorted cameras...")
        render_all(dataset.model_path, all_cameras, gaussians, pipeline, background, save_ims, op)

        # 5. 환경맵 저장 (필요 시 유지)
        env_dict = gaussians.render_env_map()
        grid = [env_dict["env1"].permute(2, 0, 1)]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env1.png"))
        
        grid = [env_dict["env2"].permute(2, 0, 1)]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env2.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images, op, True)
