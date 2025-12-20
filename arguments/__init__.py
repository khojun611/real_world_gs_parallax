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


from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        # Device Settings
        self.data_device = "cuda"
        self.eval = False

        # EnvLight Settings
        self.envmap_max_res = 512
        self.envmap_max_roughness = 0.5
        self.envmap_min_roughness = 0.08
        self.relight = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.use_asg = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Learning Rate Settings
        self.iterations = 50_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.indirect_lr = 0.0075 
        self.asg_lr = 0.0075 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        
        self.ori_color_lr = 0.0075 
        self.refl_strength_lr =  0.005 
        self.roughness_lr =  0.005 
        self.metalness_lr = 0.01
        self.uncertainty_lr = 0.001
        self.normal_lr = 0.006
        self.envmap_cubemap_lr = 0.01
        self.contrastive_lr = 0.001 # 프로젝션 헤드를 위한 학습률
        
        # Densification Settings
        self.percent_dense = 0.01

        # Regularization Parameters
        self.lambda_dssim = 0.2
        self.lambda_hybrid = 0.0
        self.lambda_hint = 0.0
        self.lambda_purity = 0.0 # 순도 loss
        #self.lambda_contrastive = 0.0  # 대조 학습 손실 가중치 (실험 시 0.1 등으로 시작)
        # <<<--- 아래 두 줄을 추가하세요. ---<<<
        #self.contrastive_num_patches = 128      # 대조 학습에 사용할 전체 패치 수
        #self.contrastive_highlight_thresh = 0.2 # 하이라이트 마스크 임계값
        #self.contrastive_patch_size = 16    # 대조 학습에 사용할 패치 크기
        # <<<--- 이 줄을 추가하거나, 기존 줄이 올바른지 확인하세요. ---<<<
        #self.contrastive_temp = 0.07        # 온도(temperature) 파라미터
        self.use_high_freq_purity_loss = False # 고주파 가중치 사용 여부
        self.purity_focal_gamma = 0.0 # Focal Loss를 위한 gamma
        self.fourier_cutoff_ratio = 0.3 # 푸리에 마스크의 컷오프 비율
        self.lambda_pseudo_normal = 0.0 # pseudo normal loss
        self.lambda_pseudo_diffuse = 0.0 # diffuse 0 (비활성화)
        self.lambda_pseudo_specular = 0.0 # specular 0 (비활성화)
        self.lambda_normal_hybrid = 0.0
        self.lambda_diffuse_cons = 0.0 # diffuse constraint
        self.lambda_specular_neutral = 0.0 # specular_neural
        # --- Specular Presence Hinge Loss 파라미터 추가 ---
        self.lambda_spec_presence = 0.0      # 손실 가중치 (실험 시 1.0 ~ 10.0 등으로 시작)
        self.spec_presence_from_iter = 3000  # 이 손실을 시작할 반복 시점
        self.spec_presence_decay_start_iter = 25000 # 가중치 감쇠 시작 시점
        self.spec_presence_decay_end_iter = 45000   # 가중치가 0이 되는 시점
        self.spec_presence_tau = 0.0         # 하이라이트로 간주할 밝기 임계값 (0~1)
        self.spec_presence_alpha = 0.0       # Specular가 최소한 가져야 할 밝기 비율 (0~1)
        
        # --- Metallic 맵 기반 손실을 위한 하이퍼파라미터 ---
        self.lambda_metallic_supervision = 0.1
        self.lambda_diffuse_metal = 0.0
        self.lambda_specular_luminance = 0.0
        self.lambda_unc_spec_consistency = 0.0  # Uncertainty-Specular 일관성 손실 가중치
        self.unc_spec_from_iter = 6500          # 이 손실을 적용 시작할 시점
        self.uncertainty_amp_gain = 20.0  # 증폭 계수(k). 클수록 대비가 강해짐.
        self.uncertainty_amp_bias = 0.0  # 기준점(b). uncertainty의 평균적인 값 근처로 설정.
        self.uncertainty_metallic_threshold = 0.0
        self.metallic_threshold = 0.9
        self.metallic_loss_from_iter = 3000
        self.lambda_roughness_metal = 0.00      # <<-- 금속 영역의 roughness 손실 가중치
        self.lambda_roughness_non_metal = 0.0  # <<-- 비금속 영역의 roughness 손실 가중치
        # ----------------------------------------------------
        # --- 여기까지 추가 ---
        # <<<--- 여기에 아래 파라미터들을 추가하세요. ---<<<
        self.lambda_energy = 0.0         # 에너지 보존 손실 가중치
        self.energy_loss_from_iter = 5000  # 이 손실을 적용 시작할 시점
        # <<<--- 여기에 아래 파라미터들을 추가하세요. ---<<<
        self.lambda_diffuse_suppress = 0.0  # Diffuse 억제 손실 가중치
        self.suppress_from_iter = 10000     # 이 손실을 적용 시작할 시점
        self.suppress_refl_thresh = 0.5     # 반사로 간주할 refl_strength 임계값
        self.suppress_unc_thresh = 0.3      # 반사로 간주할 uncertainty 임계값
        
        self.lambda_sparsity = 0.0
        self.lambda_dist = 0.0
        self.lambda_normal_render_depth = 0.05
        self.lambda_normal_smooth = 0.0
        self.lambda_depth_smooth = 0.0

        # initial values
        self.init_roughness_value = 0.1
        self.init_refl_value = 0.01
        self.init_refl_value_vol = 0.01
        self.rough_msk_thr = 0.01
        self.refl_msk_thr = 0.02
        self.refl_msk_thr_vol = 0.02

        self.enlarge_scale = 1.5
        self.train_on_all = False
        # Opacity and Densify Settings
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25000 

        # Extra settings
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.05

        # Stage Settings
        self.initial = 0
        self.init_until_iter = 0 
        self.volume_render_until_iter = 0 
        self.normal_smooth_from_iter = 0
        self.normal_smooth_until_iter = 25000
        self.uncertainty_from_iter = 3000
        self.purity_loss_from_iter = 3000
        self.diffuse_cons_from_iter = 5000
        self.specular_neutral_from_iter = 500000
        self.sparsity_loss_from_iter = 500000
        self.pseudo_normal_from_iter = 3000
        self.contrastive_from_iter = 300000   # <<<--- 이 줄을 추가하세요.
        
        self.indirect = 0
        self.indirect_from_iter =  80000 

        self.feature_rest_from_iter = 5_000
        self.normal_prop_until_iter = 25_000 

        self.normal_prop_interval = 1000
        self.opac_lr0_interval = 200
        self.densification_interval_when_prop = 500

        self.normal_loss_start = 0
        self.dist_loss_start = 3000

        # Environmental Scoping
        self.use_env_scope = False
        self.env_scope_center = [0., 0., 0.]
        self.env_scope_radius = 0.0
        
        
        
        # --- [Parallax Correction Settings] ---
        # Parallax Correction 활성화 여부 및 박스 설정
        self.use_parallax_correction = False
        # Box Min/Max: Scene 크기에 맞춰 설정 필요 (예: 방의 모서리 좌표)
        #self.env_box_min = [-3.0, -3.0, -3.0] 
        #self.env_box_max = [3.0, 3.0, 3.0]
        # 큐브맵의 중심 위치 (보통 0,0,0)
        #self.env_center = [0.0, 0.0, 0.0]
        #self.env_box_lr = 0.05
        # --------------------------------------
        # [▼▼▼ 추가된 옵션들 ▼▼▼]
        self.train_env_box = False     # True일 때만 박스 크기를 학습(Fine-tuning)함
        self.env_box_lr = 0.01          # 박스 파라미터 학습률
        self.lambda_box_reg = 0.05       # 박스 크기가 초기값에서 너무 벗어나지 않게 잡는 규제 가중치
        # --------------------------------------
        
        # SRGB Transformation
        self.srgb = False

        # mesh
        self.voxel_size = -1.0
        self.depth_trunc = -1.0
        self.sdf_trunc = -1.0
        self.mesh_res = 512
        self.num_cluster = 1

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
