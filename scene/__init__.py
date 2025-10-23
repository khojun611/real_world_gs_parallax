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
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch.nn.functional as F # <<-- 이 줄을 추가하세요
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            # --- 아래 [STEP 3] print 블록 추가 ---
            # print(f"\n[STEP 3] Right after receiving scene_info in Scene.__init__:")
            if scene_info.train_cameras:
                print(f"  - First train_camera '{scene_info.train_cameras[0].image_name}': has_material={scene_info.train_cameras[0].has_pseudo_material}")
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            # 셔플할 리스트의 복사본을 만듭니다.
            train_cams_copy = list(scene_info.train_cameras)
            random.shuffle(train_cams_copy)
            
            # _replace()를 사용해 train_cameras가 교체된 새로운 scene_info 객체를 생성합니다.
            scene_info = scene_info._replace(train_cameras=train_cams_copy)
            
            # --- 아래 [STEP 4] print 블록 추가 (셔플 로직 바로 다음) ---
            # print(f"\n[STEP 4] After shuffling in Scene.__init__:")
            if scene_info.train_cameras:
                # <<-- has_pseudo_diffuse를 has_pseudo_material로 변경 -->>
                print(f"  - First train_camera '{scene_info.train_cameras[0].image_name}': has_material={scene_info.train_cameras[0].has_pseudo_material}")

            if scene_info.test_cameras:
                test_cams_copy = list(scene_info.test_cameras)
                random.shuffle(test_cams_copy)
                scene_info = scene_info._replace(test_cameras=test_cams_copy)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if args.relight:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"), relight=True, args=args)
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"), args=args)        
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]