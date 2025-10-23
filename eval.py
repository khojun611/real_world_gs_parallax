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

def find_latest_mesh_iter(model_path):
    """Return largest N from files like <model_path>/test_000123.ply; None if none."""
    ply_list = glob.glob(os.path.join(model_path, "test_*.ply"))
    if not ply_list:
        return None
    iters = []
    for p in ply_list:
        m = re.search(r'test_(\d+)\.ply$', os.path.basename(p))
        if m:
            iters.append(int(m.group(1)))
    return max(iters) if iters else None
# ----------------------------------------


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

def render_set(model_path, views, gaussians, pipeline, background, save_ims, opt):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "test", "renders")
        color_path = os.path.join(render_path, 'rgb')
        normal_path = os.path.join(render_path, 'normal')
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)

    ssims = []
    psnrs = []
    lpipss = []
    render_times = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        t1 = time.time()
        
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        render_time = time.time() - t1
        
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0).cuda()
        gt = gt[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(lpips(render_color, gt, net_type='vgg').item())
        render_times.append(render_time)

        if save_ims:
            # Save the rendered color image
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            # Save the normal map if available
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
            
    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()
    fps = 1.0 / np.array(render_times).mean()
    print('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))

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
        gt = torch.clamp(view.original_image, 0.0, 1.0).cuda()
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
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                # 실제 존재하는 test_*.ply 중 최신 iter를 찾는다
        
        mesh_iter = find_latest_mesh_iter(dataset.model_path)
        if indirect and mesh_iter is not None:
            op.indirect = 1
            ok = gaussians.load_mesh_from_ply(dataset.model_path, mesh_iter)
            if not ok:
                print(f"[WARN] Mesh load failed at iter {mesh_iter}. Falling back to indirect=0")
                op.indirect = 0
        else:
            if indirect:
                print("[WARN] No test_*.ply found. Falling back to indirect=0")
            op.indirect = 0

        
        # render_set_train(dataset.model_path, scene.getTrainCameras(), gaussians, pipeline, background, save_ims, op)
        render_set(dataset.model_path, scene.getTestCameras(), gaussians, pipeline, background, save_ims, op)
        
        env_dict = gaussians.render_env_map()
        grid = [
            env_dict["env1"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env1.png"))
        grid = [
            env_dict["env2"].permute(2, 0, 1),
        ]
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
