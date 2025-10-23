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
import glob, re
import matplotlib.cm as cm
import torch.nn.functional as F

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
    """view.image_name에서 확장자 제거한 안전한 파일명 반환"""
    base = os.path.basename(str(name))
    stem, _ = os.path.splitext(base)
    # 공백/이상문자 정리 (선택)
    return stem.replace(' ', '_')

def render_set(model_path, views, gaussians, pipeline, background, save_ims, opt):
    import matplotlib.cm as cm
    import torch.nn.functional as F

    def _apply_colormap(tensor_gray, H, W, cmap='jet', vmin=0.0, vmax=1.0):
        """1채널 텐서를 [3,H,W]로 jet 등 컬러맵 적용."""
        if tensor_gray is None or tensor_gray.numel() == 0:
            return torch.zeros(3, H, W, device='cuda')
        x = tensor_gray.detach().squeeze().float().clamp(vmin, vmax)
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        x = x.unsqueeze(0).unsqueeze(0)  # [1,1,h,w] 가정
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False).squeeze()
        cm_np = cm.get_cmap(cmap)(x.detach().cpu().numpy())[:, :, :3]  # [H,W,3]
        return torch.from_numpy(cm_np).permute(2, 0, 1).to(tensor_gray.device)  # [3,H,W]

    def _safe_stem(name, fallback):
        # view.image_name이 'xxx.png' 또는 'xxx'일 수 있어 안전하게 본체만 추출
        base = os.path.basename(str(name)) if name is not None else str(fallback)
        stem, _ = os.path.splitext(base)
        return stem if len(stem) > 0 else str(fallback)

    if save_ims:
        render_path   = os.path.join(model_path, "test", "renders")
        color_path    = os.path.join(render_path, 'rgb')
        normal_path   = os.path.join(render_path, 'normal')
        base_path     = os.path.join(render_path, 'basecolor')
        diffuse_path  = os.path.join(render_path, 'diffuse')
        specular_path = os.path.join(render_path, 'specular')
        refl_path     = os.path.join(render_path, 'reflection')  # jet 컬러맵
        rough_path    = os.path.join(render_path, 'roughness')   # jet 컬러맵
        os.makedirs(color_path, exist_ok=True)
        os.makedirs(normal_path, exist_ok=True)
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(diffuse_path, exist_ok=True)
        os.makedirs(specular_path, exist_ok=True)
        os.makedirs(refl_path, exist_ok=True)
        os.makedirs(rough_path, exist_ok=True)

    ssims, psnrs, lpipss, render_times = [], [], [], []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # eval 시 mask off
        t1 = time.time()
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        render_time = time.time() - t1
        render_times.append(render_time)

        # metrics
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)[None]    # [1,3,H,W]
        gt           = torch.clamp(view.original_image, 0.0, 1.0).cuda()[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        try:
            lpipss.append(lpips(render_color, gt, net_type='vgg').item())
        except Exception:
            # lpips 모듈 미초기화/미설치 등인 경우 0으로 채움
            lpipss.append(0.0)

        if save_ims:
            stem = _safe_stem(getattr(view, "image_name", None), f"{idx:05d}")

            # RGB
            torchvision.utils.save_image(render_color, os.path.join(color_path, f'{stem}.png'))

            # Normal
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, f'{stem}.png'))

            # Base Color (Albedo)
            if 'base_color_map' in rendering:
                basecolor = torch.clamp(rendering['base_color_map'], 0.0, 1.0)
                torchvision.utils.save_image(basecolor, os.path.join(base_path, f'{stem}.png'))

            # Diffuse
            if 'diffuse_map' in rendering:
                diffuse = torch.clamp(rendering['diffuse_map'], 0.0, 1.0)
                torchvision.utils.save_image(diffuse, os.path.join(diffuse_path, f'{stem}.png'))

            # Specular
            if 'specular_map' in rendering:
                specular = torch.clamp(rendering['specular_map'], 0.0, 1.0)
                torchvision.utils.save_image(specular, os.path.join(specular_path, f'{stem}.png'))

            # Reflection Strength (jet)
            if 'refl_strength_map' in rendering:
                refl = torch.clamp(rendering['refl_strength_map'], 0.0, 1.0)
                H, W = render_color.shape[-2:]
                refl_jet = _apply_colormap(refl, H, W, cmap='jet', vmin=0.0, vmax=1.0)
                torchvision.utils.save_image(refl_jet, os.path.join(refl_path, f'{stem}.png'))

            # Roughness (jet)
            if 'roughness_map' in rendering:
                rough = torch.clamp(rendering['roughness_map'], 0.0, 1.0)
                H, W = render_color.shape[-2:]
                rough_jet = _apply_colormap(rough, H, W, cmap='jet', vmin=0.0, vmax=1.0)
                torchvision.utils.save_image(rough_jet, os.path.join(rough_path, f'{stem}.png'))

    # summary
    ssim_v = float(np.mean(ssims)) if ssims else 0.0
    psnr_v = float(np.mean(psnrs)) if psnrs else 0.0
    lpip_v = float(np.mean(lpipss)) if lpipss else 0.0
    fps    = 1.0 / float(np.mean(render_times)) if render_times else 0.0
    print(f'psnr:{psnr_v}, ssim:{ssim_v}, lpips:{lpip_v}, fps:{fps}')
    with open(os.path.join(model_path, 'metric.txt'), 'w') as f:
        f.write(f'psnr:{psnr_v}, ssim:{ssim_v}, lpips:{lpip_v}, fps:{fps}')


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op, indirect):
    with torch.no_grad():
        print(f"[INFO] model_path={dataset.model_path}")
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f"[INFO] white_background={dataset.white_background} -> bg={bg_color}")

        # indirect mesh
        mesh_iter = find_latest_mesh_iter(dataset.model_path)
        print(f"[INFO] latest test_*.ply iter: {mesh_iter}")
        if indirect and mesh_iter is not None:
            op.indirect = 1
            ok = gaussians.load_mesh_from_ply(dataset.model_path, mesh_iter)
            print(f"[INFO] load_mesh_from_ply({mesh_iter}) -> {ok}")
            if not ok:
                print(f"[WARN] Mesh load failed. Falling back to indirect=0")
                op.indirect = 0
        else:
            if indirect:
                print("[WARN] No test_*.ply found. Falling back to indirect=0")
            op.indirect = 0

        train_cams = scene.getTrainCameras()
        test_cams  = scene.getTestCameras()
        print(f"[INFO] #train_cameras={len(train_cams)}, #test_cameras={len(test_cams)}")

        render_set(dataset.model_path, test_cams, gaussians, pipeline, background, save_ims, op)

        # env map dump (있을 때만)
        try:
            env_dict = gaussians.render_env_map()
            if "env1" in env_dict and env_dict["env1"] is not None:
                grid = [env_dict["env1"].permute(2, 0, 1)]
                grid = make_grid(grid, nrow=1, padding=10)
                save_image(grid, os.path.join(dataset.model_path, "env1.png"))
                print("[INFO] saved env1.png")
            if "env2" in env_dict and env_dict["env2"] is not None:
                grid = [env_dict["env2"].permute(2, 0, 1)]
                grid = make_grid(grid, nrow=1, padding=10)
                save_image(grid, os.path.join(dataset.model_path, "env2.png"))
                print("[INFO] saved env2.png")
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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images, op, True)
