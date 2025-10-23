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

from scene.cameras import Camera
import numpy as np
import os, cv2, torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = float(resolution_scale * args.resolution)
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    HWK = None
    if cam_info.K is not None:
        K = cam_info.K.copy()
        K[:2] = K[:2] / scale
        HWK = (resolution[1], resolution[0], K)

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
    
    refl_msk = None # 이 부분은 필요 시 유지

     # --- << Material map 처리로 로직 변경 >> ---
    resized_material_map_np = None
    if cam_info.has_pseudo_material and cam_info.pseudo_material_map is not None:
        # <<-- 1. NumPy 배열을 float32 타입으로 강제 변환합니다. -->>
        material_map_float32 = cam_info.pseudo_material_map.astype(np.float32)
        
        # <<-- 2. 변환된 배열을 사용하여 리사이즈를 수행합니다. -->>
        resized_material_map_np = cv2.resize(material_map_float32, 
                                             (resolution[0], resolution[1]),
                                             interpolation=cv2.INTER_LINEAR)
    # ----------------------------------------
    resized_normal_map = None
    if cam_info.has_normal:
        normal_map_np = cam_info.normal_map
        resized_normal_map_np = cv2.resize(normal_map_np, 
                                           (resolution[0], resolution[1]),
                                           interpolation=cv2.INTER_LINEAR)
        norm = np.linalg.norm(resized_normal_map_np, axis=2, keepdims=True)
        resized_normal_map = resized_normal_map_np / (norm + 1e-8)
    
    # --- << Camera 객체 생성자 수정 >> ---
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, 
                  # <<-- 이 두 줄을 추가하여 해상도 값을 전달합니다.
                  image_height=resolution[1], image_width=resolution[0],
                  data_device=args.data_device, HWK=HWK, gt_refl_mask=refl_msk,
                  normal_map=resized_normal_map,
                  has_normal=cam_info.has_normal,
                  pseudo_material_map=resized_material_map_np,
                  has_pseudo_material=cam_info.has_pseudo_material)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry