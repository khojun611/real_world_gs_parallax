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

import os
import torch
import open3d as o3d
from random import randint
from utils.loss_utils import calculate_loss, l1_loss
from gaussian_renderer import render_surfel, render_initial, render_volume, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth
from utils.graphics_utils import linear_to_srgb
from utils.mesh_utils import GaussianExtractor, post_process_mesh
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import matplotlib.cm as cm



def apply_colormap(tensor_gray, H, W, cmap='jet', vmin=0.0, vmax=1.0):
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

# --- 2단계: 불확실성 전용 Fine-tuning 함수 ---
def fine_tune_uncertainty_only(dataset, opt, pipe, checkpoint, finetune_iter):
    print("[INFO] Starting Stage 2: Uncertainty-only fine-tuning.")
    
    gaussians = GaussianModel(dataset.sh_degree)
    
    # <<-- 핵심 수정: 체크포인트 경로를 기반으로 원본 모델 경로를 올바르게 설정합니다. -->>
    # 1. 원본 모델 경로는 체크포인트 파일이 있는 디렉토리입니다.
    source_model_path = os.path.dirname(checkpoint)
    print(f"[INFO] Source model path identified as: {source_model_path}")
    
    # 2. Scene 객체를 초기화할 때, 원본 모델 경로를 사용하도록 dataset 객체를 임시로 수정합니다.
    #    이렇게 하면 .ply 파일과 cameras.json을 올바른 위치에서 찾을 수 있습니다.
    original_dataset_model_path = dataset.model_path 
    dataset.model_path = source_model_path
    
    print(f"[INFO] Loading checkpoint from: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint)

    print(f"[INFO] Initializing scene with data from iteration {first_iter} to load trained envmap...")
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False)
    
    # 3. Scene 객체가 생성된 후, 저장 경로는 다시 사용자가 -m으로 지정한 출력 경로로 설정합니다.
    scene.model_path = original_dataset_model_path
    os.makedirs(scene.model_path, exist_ok=True) 
    print(f"[INFO] Output path for fine-tuned model set to: {scene.model_path}")
    
    gaussians.restore(model_params, opt)
    print(f"[INFO] Model and EnvMap fully restored from iteration {first_iter}.")

    print("[INFO] Sanitizing all loaded Gaussian parameters...")
    with torch.no_grad():
        for name, attr in vars(gaussians).items():
            if isinstance(attr, torch.nn.Parameter):
                if torch.is_tensor(attr.data) and attr.numel() > 0:
                    attr.data = torch.nan_to_num(attr.data, nan=0.0, posinf=0.0, neginf=0.0)
        if gaussians._rotation.numel() > 0:
            gaussians._rotation.data = F.normalize(gaussians._rotation.data, p=2, dim=1)
    print("[INFO] Sanitizing complete.")

    print("[INFO] Freezing main model parameters...")
    for name, attr in vars(gaussians).items():
        if isinstance(attr, torch.nn.Parameter):
            if "_uncertainty" not in name:
                attr.requires_grad = False
            else:
                print(f"[INFO] Activating gradient for: {name}")

    optimizer_uc = torch.optim.Adam(
        [{'params': gaussians._uncertainty, 'lr': opt.uncertainty_lr, "name": "uncertainty"}]
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    
    progress_bar = tqdm(range(1, finetune_iter + 1), desc="Uncertainty Fine-tuning")

    for iteration in progress_bar:
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_surfel(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)

        opt.uncertainty_from_iter = 0 
        loss, tb_dict = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        
        if not torch.isnan(loss):
            loss.backward()
            optimizer_uc.step()
        
        optimizer_uc.zero_grad()
        
        if iteration % 10 == 0:
            nll_loss_val = tb_dict.get("loss_nll", 0.0)
            progress_bar.set_postfix({"NLL Loss": f"{nll_loss_val:.{5}f}"})

    final_iter = first_iter + finetune_iter
    print(f"\n[INFO] Uncertainty training complete. Saving final model at iteration {final_iter}.")
    scene.save(final_iter)
    
    torch.save((gaussians.capture(), final_iter), scene.model_path + f"/chkpnt{final_iter}_uc.pth")
    print(f"[INFO] Final model saved to {scene.model_path}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, model_path, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    # Set up parameters 
    TOT_ITER = opt.iterations + 1
    TEST_INTERVAL = 1000
    MESH_EXTRACT_INTERVAL = 2000

    # For real scenes
    USE_ENV_SCOPE = opt.use_env_scope  # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4


    gaussians = GaussianModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter)) # #
    scene = Scene(dataset, gaussians)  # init all parameters(pos, scale, rot...) from pcds
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_initial, pipe, bg_color=bg_color) 

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_normal_smooth_for_log = 0.0
    ema_depth_smooth_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print(f'Propagation until: {opt.normal_prop_until_iter }')
    print(f'Densify until: {opt.densify_until_iter}')
    print(f'Total iterations: {TOT_ITER}')


    initial_stage = opt.initial
    if not initial_stage:
        opt.init_until_iter = 0


    # Training loop
    while iteration < TOT_ITER:
        iter_start.record()

        gaussians.update_learning_rate(iteration)


        # Increase SH levels every 1000 iterations
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Control the init stage
        if iteration > opt.init_until_iter:
            initial_stage = False
        
        # Control the indirect stage
        if iteration == opt.indirect_from_iter + 1:
            opt.indirect = 1


        if iteration == (opt.volume_render_until_iter + 1) and opt.volume_render_until_iter > opt.init_until_iter:
            reset_gaussian_para(gaussians, opt)

        # Initialize envmap
        if not initial_stage:
            if iteration <= opt.volume_render_until_iter:
                envmap2 = gaussians.get_envmap_2 
                if envmap2 is not None: envmap2.build_mips()
            else:
                envmap = gaussians.get_envmap 
                if envmap is not None: envmap.build_mips()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


        # Set render
        render = select_render_method(iteration, opt, initial_stage)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        total_loss, tb_dict = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        dist_loss, normal_loss, loss, Ll1, normal_smooth_loss, depth_smooth_loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"], tb_dict["loss_normal_smooth"], tb_dict["loss_depth_smooth"] 

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            total_loss += REFL_MSK_LOSS_W * refl_msk_loss
        
        total_loss.backward()

        iter_end.record()


        with torch.no_grad():
            
            if iteration % TEST_INTERVAL == 0 or iteration == first_iter + 1 or iteration == opt.volume_render_until_iter + 1:
                save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration, initial_stage, tb_dict)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            ema_normal_smooth_for_log = 0.4 * normal_smooth_loss + 0.6 * ema_normal_smooth_for_log
            ema_depth_smooth_for_log = 0.4 * depth_smooth_loss + 0.6 * ema_depth_smooth_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            if iteration % TEST_INTERVAL == 0:
                psnr_test = evaluate_psnr(scene, render, {"pipe": pipe, "bg_color": background, "opt": opt})
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, {"pipe": pipe, "bg_color": background, "opt":opt})

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration <= opt.init_until_iter:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= opt.normal_prop_until_iter :
                    opacity_reset_intval = 3000
                    densification_interval = opt.densification_interval_when_prop
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
                                                size_threshold)

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity0()
                    gaussians.reset_refl(exclusive_msk=outside_msk)
                if opt.opac_lr0_interval > 0 and (
                        opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.opac_lr0_interval == 0:
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if (opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.normal_prop_interval == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        if iteration > opt.volume_render_until_iter and opt.volume_render_until_iter > opt.init_until_iter:
                            gaussians.dist_color(exclusive_msk=outside_msk)
                            # gaussians.dist_albedo(exclusive_msk=outside_msk)

                        gaussians.reset_scale(exclusive_msk=outside_msk)
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter :
                            gaussians.set_opacity_lr(0.0)
                
            if (iteration >= opt.indirect_from_iter and iteration % MESH_EXTRACT_INTERVAL == 0) or iteration == (opt.indirect_from_iter):
                if not HAS_RESET0:
                    gaussExtractor.reconstruction(scene.getTrainCameras())
                    if 'ref_real' in dataset.source_path:
                        mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    else:
                        depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                        voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                        sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                    ply_path = os.path.join(model_path,f'test_{iteration:06d}.ply')
                    o3d.io.write_triangle_mesh(ply_path, mesh)
                    gaussians.update_mesh(mesh)

            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

        iteration += 1

# ============================================================
# Utils for training

def select_render_method(iteration, opt, initial_stage):
    if initial_stage:
        render = render_initial
    elif iteration <= opt.volume_render_until_iter:
        render = render_volume
    else:  
        render = render_surfel
    return render

def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage, tb_dict):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)
        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])
        H, W = error_map.shape[-2:]

        if initial_stage:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),
                render_pkg["render"], 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg.get("surf_depth", torch.empty(0, device='cuda'))),  
                render_pkg.get("rend_normal", torch.zeros_like(error_map)) * 0.5 + 0.5, 
                render_pkg.get("surf_normal", torch.zeros_like(error_map)) * 0.5 + 0.5, 
                error_map 
            ]
        elif iteration <= opt.volume_render_until_iter:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"], 
                render_pkg.get("base_color_map", torch.zeros_like(error_map)), 
                render_pkg.get("diffuse_map", torch.zeros_like(error_map)),     
                render_pkg.get("specular_map", torch.zeros_like(error_map)),  
                apply_colormap(render_pkg.get("refl_strength_map", torch.empty(0, device='cuda')), H, W),  
                apply_colormap(render_pkg.get("roughness_map", torch.empty(0, device='cuda')), H, W),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg.get("surf_depth", torch.empty(0, device='cuda'))), 
                render_pkg.get("rend_normal", torch.zeros_like(error_map)) * 0.5 + 0.5,  
                render_pkg.get("surf_normal", torch.zeros_like(error_map)) * 0.5 + 0.5, 
                error_map
            ]
            if "rgb_uncertainty_map" in render_pkg:
                vis_uncertainty = apply_colormap(render_pkg["rgb_uncertainty_map"], H, W, cmap='viridis', vmax=0.5)
                visualization_list.insert(5, vis_uncertainty)
            
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"],
                    render_pkg["indirect_light"],
                ]
        else:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"],  
                render_pkg.get("base_color_map", torch.zeros_like(error_map)),  
                render_pkg.get("diffuse_map", torch.zeros_like(error_map)),
                render_pkg.get("specular_map", torch.zeros_like(error_map)),
                apply_colormap(render_pkg.get("rgb_uncertainty_map", torch.empty(0, device='cuda')), H, W, cmap='viridis', vmax=0.5),
                apply_colormap(render_pkg.get("refl_strength_map", torch.empty(0, device='cuda')), H, W),  
                apply_colormap(render_pkg.get("roughness_map", torch.empty(0, device='cuda')), H, W),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg.get("surf_depth", torch.empty(0, device='cuda'))),  
                render_pkg.get("rend_normal", torch.zeros_like(error_map)) * 0.5 + 0.5,  
                render_pkg.get("surf_normal", torch.zeros_like(error_map)) * 0.5 + 0.5,  
                error_map, 
            ]
            # <<-- 핵심 수정: 안전한 키 접근 방식으로 변경 -->>
            vis_chroma = tb_dict.get("vis_chroma_loss_map")
            if vis_chroma is not None:
                print("chroma loss2")
                vis_chroma = apply_colormap(vis_chroma, H, W, cmap='magma', vmax=0.1)
                visualization_list.append(vis_chroma)

            vis_hf_mask = tb_dict.get("vis_high_freq_mask")
            if vis_hf_mask is not None:
                print("high freq2")
                vis_hf_mask = apply_colormap(vis_hf_mask, H, W, cmap='gray')
                visualization_list.append(vis_hf_mask)

        grid = make_grid(visualization_list, nrow=5)
        scale = grid.shape[-2] / 800
        if scale > 1:
            grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
                env_dict = gaussians.render_env_map_2() 
            else:
                env_dict = gaussians.render_env_map()

            if "env1" in env_dict and env_dict["env1"] is not None:
                grid = [env_dict["env1"].permute(2, 0, 1)]
                grid = make_grid(grid, nrow=1, padding=10)
                save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderkwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs):
    psnr_test = 0.0
    torch.cuda.empty_cache()
    if len(scene.getTestCameras()):
        for viewpoint in scene.getTestCameras():
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(scene.getTestCameras())
        
    torch.cuda.empty_cache()
    return psnr_test





# ============================================================================
# Main function


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
     # --- 3. 새로운 인자 추가 ---
    parser.add_argument("--finetune_uncertainty", type=str, default=None, help="Path to a checkpoint to start uncertainty-only fine-tuning.")
    parser.add_argument("--finetune_iter", type=int, default=5000, help="Number of iterations for fine-tuning.")
    # (기존 다른 인자들은 그대로 유지)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    args.test_iterations.append(args.volume_render_until_iter)

    
    if not args.model_path:
        # 获取当前时间并格式化为精确到分钟
        current_time = datetime.now().strftime('%m%d_%H%M')
        # 获取args.source_path的最后一个子目录名
        last_subdir = os.path.basename(os.path.normpath(args.source_path))

        
        # 生成带有时间戳和opt属性的简洁输出目录
        args.model_path = os.path.join(
            "./output/", f"{last_subdir}/",
            f"{last_subdir}-{current_time}"
        )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # --- 4. 모드 분기 로직 추가 ---
    if args.finetune_uncertainty:
        # 2단계: 불확실성 fine-tuning 모드 실행
        # <<-- 핵심 수정: args.finetune_iter를 함수에 전달합니다. -->>
        fine_tune_uncertainty_only(lp.extract(args), op.extract(args), pp.extract(args), args.finetune_uncertainty, args.finetune_iter)
    else:
        # 1단계: 일반 학습 모드 실행
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.model_path)

    print("\nProcess complete.")