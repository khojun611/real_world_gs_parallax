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
from scene.light2 import EnvLight

def render_all(model_path, views, gaussians, pipeline, background, save_ims, opt):
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
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        #lpipss.append(lpips(render_color, gt, net_type='vgg').item())
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
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        
        # --- ⬇️ 재조명을 위한 코드 추가 ⬇️ ---
        print("Overwriting environment map for relighting...")
        # 1. 여기에 새로운 .hdr 파일의 경로를 입력하세요.
        new_hdr_path = "814-hdri-skies-com.hdr" 
        
        # 2. 새로운 환경맵 객체를 생성합니다.
        new_env_map = EnvLight(path=new_hdr_path, device='cuda', trainable=False).cuda()
        new_env_map.build_mips() # PBR 렌더링을 위해 Mipmap을 생성합니다.
        
        # 3. 학습된 환경맵을 새로운 환경맵으로 덮어씁니다.
        gaussians.env_map = new_env_map
        gaussians.env_map_2 = new_env_map # render_volume과의 호환성을 위해 둘 다 교체
        # --- ⬆️ 재조명 코드 끝 ⬆️ ---

        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        if indirect:
            op.indirect = 1
            gaussians.load_mesh_from_ply(dataset.model_path, iteration)

        print("Combining train and test cameras...")
        combined_cameras = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
        
        unique_cameras = {cam.uid: cam for cam in combined_cameras}
        all_cameras = list(unique_cameras.values())
        all_cameras.sort(key=lambda x: x.image_name)
        
        print(f"Total unique cameras to render: {len(all_cameras)}")
        
        print("Rendering all sorted cameras with new lighting...")
        render_all(dataset.model_path, all_cameras, gaussians, pipeline, background, save_ims, op)

        # 4. (선택 사항) 교체된 환경맵을 이미지로 저장하여 확인합니다.
        env_dict = gaussians.render_env_map()
        grid = [env_dict["env1"].permute(2, 0, 1)]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "relit_env.png"))
        
        


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
