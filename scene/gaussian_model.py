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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
# from utils.point_utils import addpoint, combine_pointcloud, downsample_point_cloud_open3d, find_indices_in_A
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness
from typing import Literal
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        # self._deformation =  torch.empty(0)
        self._deformation = deform_network(args)
        # self.grid = TriPlaneGrid()
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.setup_functions()
        self._language_feature = None
        
        # CoTracker3 control point support (optional)
        self._is_control_point_driven = None  # (N_gaussians,) boolean mask
        self._control_point_positions_precomputed = None  # (T, N_gaussians, 3) or None, precomputed positions for control-point-driven Gaussians
        self._num_frames = None  # Number of frames (T) for time-to-frame conversion
        self._cotracker_data = None  # dict with control point data (set in Scene.__init__)

    def capture(self, include_feature=False):
        if include_feature:
            assert self._language_feature is not None, "没有设置language feature"
            return (
                self.active_sh_degree,
                self._xyz,
                self._deformation.state_dict(),
                self._deformation_table,
                # self.grid,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._language_feature,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._deformation.state_dict(),
                self._deformation_table,
                # self.grid,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
    
    def restore(self, model_args, training_args, mode='train',stage='lang-fine',joint_train=False,no_dlang=False,init_from_stage='fine-lang',coarse_freeze_xyz=False):
        if len(model_args) == 15:
            (self.active_sh_degree, 
            self._xyz, 
            deform_state,
            self._deformation_table,
            
            # self.grid,
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._language_feature,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif len(model_args) == 14:
            (self.active_sh_degree, 
            self._xyz, 
            deform_state,
            self._deformation_table,
            
            # self.grid,
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            if not training_args.include_feature: # 如果是以原始gs为初始化来训练feature的话，就不需要restore optimizer
                self.optimizer.load_state_dict(opt_dict)

            self._deformation.load_state_dict(deform_state)
        if mode == 'train':
            self.training_setup(training_args,stage,joint_train,no_dlang,init_from_stage=init_from_stage,coarse_freeze_xyz=coarse_freeze_xyz)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_language_feature(self):
        if self._language_feature is not None:
            return self._language_feature
        else:
            raise ValueError('没有设置language feature')
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        # # Initialize features to zero (no color initialization from point cloud)
        # features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, 3:, 1:] = 0.0

        # Initialize colors from point cloud
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        # self.grid = self.grid.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        
        # Initialize CoTracker control-point-driven mask if data is available
        if hasattr(self, '_cotracker_data') and self._cotracker_data is not None:
            from utils.cotracker_gaussian_utils import initialize_control_point_driven_mask
            n_gaussians = fused_point_cloud.shape[0]
            gaussian_control_point_indices = self._cotracker_data["gaussian_control_point_indices"]
            self._is_control_point_driven = initialize_control_point_driven_mask(
                n_gaussians, gaussian_control_point_indices
            ).cuda()
            
            # Load precomputed positions (already torch tensor)
            gaussian_positions_precomputed = self._cotracker_data["gaussian_positions_precomputed"]
            self._control_point_positions_precomputed = gaussian_positions_precomputed.float().cuda()  # (T, N_gaussians, 3)
            print(f"shape of self._control_point_positions_precomputed: {self._control_point_positions_precomputed.shape}")
        else:
            self._is_control_point_driven = None
            self._control_point_positions_precomputed = None
    def training_setup(self, training_args,stage:Literal['coarse-base','coarse-lang','fine-base','fine-lang','fine-lang-discrete'],joint_train=False,no_dlang=False,init_from_stage='fine-lang',coarse_freeze_xyz=False):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")

        # Determine if xyz should be frozen (only in coarse stages when flag is set)
        # This allows training static appearance/features on a fixed geometry from high-quality depth initialization
        freeze_xyz = coarse_freeze_xyz and "coarse" in stage
        
        # Determine which Gaussians have optimizable xyz (not control-point-driven)
        if self._is_control_point_driven is not None:
            xyz_optimizable_mask = ~self._is_control_point_driven  # Inverse: True for optimizable, False for control-point-driven
        else:
            # If no CoTracker data, all Gaussians are optimizable
            xyz_optimizable_mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")

        if training_args.include_feature and ("lang" in stage):
            if 'discrete' in stage and self._language_feature.shape[-1]==int(os.getenv("language_feature_hiddendim",3)):
                multi_language_features = self.generate_multi_feature_centers(init_from_stage=init_from_stage)
                multi_language_features = multi_language_features.to("cuda")
                multi_language_features = multi_language_features.permute(0, 2, 1).reshape(self.get_xyz.shape[0],-1)
                self._language_feature = nn.Parameter(multi_language_features.requires_grad_(True))
            if self._language_feature is None or self._language_feature.shape[0] != self._xyz.shape[0]:
                language_feature = torch.zeros((self._xyz.shape[0], int(os.getenv("language_feature_hiddendim",3))), device="cuda")
                self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
            
            print(f"training_args.language_feature_lr:{training_args.language_feature_lr}")
            if joint_train:
                l = []
                # Only optimize xyz if not frozen AND if Gaussian is optimizable (not control-point-driven)
                if not freeze_xyz and xyz_optimizable_mask.any():
                    # For control-point-driven Gaussians, xyz is not optimized
                    # We still add _xyz to optimizer but will set requires_grad=False for control-point-driven ones
                    l.append({'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"})
                l.extend([
                    {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                    {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                    {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                ])
            else:
                l = []
            if 'fine' in stage:
                l.append(
                    {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"}
                )
                l.append(
                    {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"}
                )

            l.append({'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"})

            # Set requires_grad for all parameters
            self._xyz.requires_grad_(joint_train and not freeze_xyz)
            self._deformation.requires_grad_(joint_train)
            # Enable gradients for the actual language deformation heads (ModuleList) and their scaling alphas
            self._deformation.deformation_net.lang_deforms.requires_grad_(no_dlang==0)
            for alpha in self._deformation.deformation_net.lang_alphas:
                alpha.requires_grad_(no_dlang==0)
            # Legacy single head (used with use_discrete_lang_f only)
            self._deformation.deformation_net.lang_deform.requires_grad_(no_dlang==0)
            if 'discrete' in stage:
                self._deformation.deformation_net.discrete_coff_generator.requires_grad_(True)

            print(f"set lang_deforms.requires_grad_ to {no_dlang==0}")
            self._features_dc.requires_grad_(joint_train)
            self._features_rest.requires_grad_(joint_train)
            self._scaling.requires_grad_(joint_train)
            self._rotation.requires_grad_(joint_train)
            self._opacity.requires_grad_(joint_train)
            self._language_feature.requires_grad_(True)

        else:
            l = []
            # Only optimize xyz if not frozen
            if not freeze_xyz:
                l.append({'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"})
            l.extend([
                {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
                {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ])
            if self._language_feature is not None:
                l.append(
                    {'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"}
                )
            
            # Set requires_grad for all parameters
            # Control-point-driven Gaussians: xyz gradients are stopped in renderer by replacing with detached positions
            self._xyz.requires_grad_(not freeze_xyz)
            self._deformation.requires_grad_(True)

            self._features_dc.requires_grad_(True)
            self._features_rest.requires_grad_(True)
            self._scaling.requires_grad_(True)
            self._rotation.requires_grad_(True)
            self._opacity.requires_grad_(True)
            if self._language_feature is not None:
                self._language_feature.requires_grad_(False)
    
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15) 
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        # Language features may be absent in base-only runs; fall back to env dim
        if self._language_feature is not None:
            lang_dim = self._language_feature.shape[1]
        else:
            lang_dim = int(os.getenv("language_feature_hiddendim", 3))
        for i in range(lang_dim):
            l.append('f_lang_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    def compute_deformation(self,time):
        
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")

        self._deformation.load_state_dict(weight_dict,strict=False)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # If language features are absent (e.g., base-only runs), write zeros with env-specified dim
        if self._language_feature is None:
            lang_dim = int(os.getenv("language_feature_hiddendim", 3))
            f_lang = np.zeros((xyz.shape[0], lang_dim), dtype=np.float32)
        else:
            f_lang = self._language_feature.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, f_lang, opacities, scale, rotation), axis=1)
        # import pdb; pdb.set_trace()
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # import pdb; pdb.set_trace()
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        f_lang_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_lang_")]
        f_lang_names = sorted(f_lang_names, key = lambda x: int(x.split('_')[-1]))
        f_lang = np.zeros((xyz.shape[0], len(f_lang_names)))
        for idx, attr_name in enumerate(f_lang_names):
            f_lang[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._language_feature = nn.Parameter(torch.tensor(f_lang, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, train_language_feature=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                if group["name"] == 'language_feature':
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(train_language_feature)))
                else:
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == 'language_feature':
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(train_language_feature))
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, stage):
        valid_points_mask = ~mask
        if "lang" in stage:
            train_language_feature=True
        else:
            train_language_feature=False

        optimizable_tensors = self._prune_optimizer(valid_points_mask, train_language_feature)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self._language_feature is not None:
            self._language_feature = optimizable_tensors["language_feature"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict, train_language_feature=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                if group["name"] == 'language_feature':
                    # import pdb; pdb.set_trace()
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(train_language_feature))
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == 'language_feature':
                    # import pdb; pdb.set_trace()
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(train_language_feature))
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table, new_language_feature, stage):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        # "deformation": new_deformation
       }
        if "lang" in  stage:
            train_language_feature = True
        else:
            train_language_feature = False

        if new_language_feature is not None:
            d["language_feature"] = new_language_feature

        optimizable_tensors = self.cat_tensors_to_optimizer(d, train_language_feature)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if new_language_feature is not None:
            self._language_feature = optimizable_tensors["language_feature"]
        # self._deformation = optimizable_tensors["deformation"]
        
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, stage='coarse-base'):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # breakpoint()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        if 'fine' in  stage and self._language_feature is not None:
            new_language_feature= self._language_feature[selected_pts_mask].repeat(N,1)
        else:
            new_language_feature = None
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table,new_language_feature, stage)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter,stage)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # breakpoint()        
        new_xyz = self._xyz[selected_pts_mask] 
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        # import pdb; pdb.set_trace()
        if 'fine' in stage and self._language_feature is not None:
            new_language_feature = self._language_feature[selected_pts_mask]  
        else:
            new_language_feature = None
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table,new_language_feature, stage)

    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    def get_displayment(self,selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point<xyz_max 
        mask_b = final_point>xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]
    
        return final_point, mask_d    
    def add_point_by_mask(self, selected_pts_mask, perturb=0, stage='coarse-base'):
        selected_xyz = self._xyz[selected_pts_mask] 
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)
        # displacements = torch.randn(selected_xyz.shape[0], 3).to(self._xyz) * perturb

        # new_xyz = selected_xyz + displacements
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]
        if 'fine' in  stage and self._language_feature is not None: 
            new_language_feature = self._language_feature[selected_pts_mask][mask]
        else:
            new_language_feature = None
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table,new_language_feature, stage)
        return selected_xyz, new_xyz
    def downsample_point(self, point_cloud):
        if not hasattr(self,"voxel_size"):
            self.voxel_size = 8  
        point_downsample = point_cloud
        flag = False 
        while point_downsample.shape[0]>1000:
            if flag:
                self.voxel_size+=8
            point_downsample = downsample_point_cloud_open3d(point_cloud,voxel_size=self.voxel_size)
            flag = True
        print("point size:",point_downsample.shape[0])
        # downsampled_point_mask = torch.eq(point_downsample.view(1,-1,3), point_cloud.view(-1,1,3)).all(dim=1)
        downsampled_point_index = find_indices_in_A(point_cloud, point_downsample)
        downsampled_point_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool).to(point_downsample.device)
        downsampled_point_mask[downsampled_point_index]=True
        return downsampled_point_mask
    def grow(self, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        if not hasattr(self,"voxel_size"):
            self.voxel_size = 8  
        if not hasattr(self,"density_threshold"):
            self.density_threshold = density_threshold
        if not hasattr(self,"displacement_scale"):
            self.displacement_scale = displacement_scale
        flag = False
        point_cloud = self.get_xyz.detach().cpu()
        point_downsample = point_cloud.detach()
        downsampled_point_index = self.downsample_point(point_downsample)


        _, low_density_points, new_points, low_density_index = addpoint(point_cloud[downsampled_point_index],density_threshold=self.density_threshold,displacement_scale=self.displacement_scale,iter_pass=0)
        if new_points.shape[0] < 100 :
            self.density_threshold /= 2
            self.displacement_scale /= 2
            print("reduce diplacement_scale to: ",self.displacement_scale)

        elif new_points.shape[0] == 0:
            print("no point added")
            return
        global_mask = torch.zeros((point_cloud.shape[0]), dtype=torch.bool)

        global_mask[downsampled_point_index] = low_density_index
        global_mask
        selected_xyz, new_xyz = self.add_point_by_mask(global_mask.to(self.get_xyz.device), self.displacement_scale, stage)
        print("point growing,add point num:",global_mask.sum())
        if model_path is not None and iteration is not None:
            point = combine_pointcloud(point_cloud, selected_xyz.detach().cpu().numpy(), new_xyz.detach().cpu().numpy())
            write_path = os.path.join(model_path,"add_point_cloud")
            os.makedirs(write_path,exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(write_path,f"iteration_{stage}{iteration}.ply"),point)
        return
    def prune(self, max_grad, min_opacity, extent, max_screen_size,stage):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask,stage)

        torch.cuda.empty_cache()
    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent, stage = stage)
    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
    
    def generate_multi_feature_centers(self,sample_num=100,init_from_stage=Literal['fine-base','fine-lang']):
        '''
            Use Kmeans method to find the discrete centers of each gaussians language features
                return: centers_tensor (num of gaussians, centers_num, feature dim)
        '''
        # import pdb; pdb.set_trace()
        
        centers_num = int(os.getenv("centers_num",3))
        if init_from_stage == 'fine-base':
            # generate language feature center init from static language features
            language_feature = self.get_language_feature
            language_feature = language_feature/ (language_feature.norm(dim=-1, keepdim=True) + 1e-9)
            multi_language_feature = torch.zeros(language_feature.shape[0], centers_num, language_feature.shape[-1]).cuda()
            for i in range(centers_num):
                noise_std=0.05
                multi_language_feature[:,i,:] = torch.normal(mean=language_feature, std=noise_std)
            # import pdb; pdb.set_trace()
            print(f"init_from_stage:{init_from_stage} Done.")
            return multi_language_feature
        else:
            language_feature_precomp_final = []
            for _ in range(sample_num):
                random_time = torch.rand_like(self.get_opacity)
                # print(random_time)
                language_feature_precomp = self.get_language_feature
                language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
                language_feature_precomp_final.append(self._deformation(self.get_xyz,self.get_scaling,self.get_rotation,self.get_opacity,self.get_features,language_feature_precomp,random_time,init_centers=True)[-1])
            res = torch.stack(language_feature_precomp_final,dim=1)
            res = res.to('cpu')
            from sklearn.cluster import KMeans
            centers = []

            from tqdm import tqdm
            for i in tqdm(range(res.shape[0])):
                data_point = res[i]  # 大小为 (10, 6)
                kmeans = KMeans(n_clusters=centers_num, random_state=0).fit(data_point.detach().numpy())
                center = kmeans.cluster_centers_
                centers.append(center)
                
            centers_tensor = torch.tensor(centers).squeeze()
            print(f"init_from_stage:{init_from_stage} Done.")
            return centers_tensor
        
    