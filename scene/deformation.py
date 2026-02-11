import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))
        
        # Multiple independent language feature heads
        self.num_lang_features = int(os.getenv("num_lang_features", 2))  # default: patch + instance
        self.lang_feature_dim = int(os.getenv("lang_feature_dim", 3))   # dimension of each feature
        self.lang_deform_width = int(os.getenv("lang_deform_width", str(self.W)))  # default: same as backbone
        language_feature_hiddendim = int(os.getenv("language_feature_hiddendim", 6))  # total dim for backward compat
        
        # Create independent deformation head for each language feature
        # Each head takes its own feature + time positional encoding as input
        lang_W = self.lang_deform_width
        self.lang_deforms = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.args.timebase_pe*2+1+self.lang_feature_dim, lang_W),
                nn.ReLU(),
                nn.Linear(lang_W, lang_W),
                nn.ReLU(),
                nn.Linear(lang_W, self.lang_feature_dim)
            )
            for _ in range(self.num_lang_features)
        ])
        
        # Legacy single head for backward compatibility (used with use_discrete_lang_f)
        self.lang_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.timebase_pe*2+1+language_feature_hiddendim,self.W),nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, language_feature_hiddendim))
        self.discrete_coff_generator = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, int(os.getenv("centers_num",3))))
        
        # ReZero scaling parameters - initialize to 0 so deltas start at 0
        # This ensures the network starts as identity and gradually learns deformations
        if getattr(self.args, 'rezero_init', False):
            self.pos_alpha = nn.Parameter(torch.zeros(1))
            self.scales_alpha = nn.Parameter(torch.zeros(1))
            self.rotations_alpha = nn.Parameter(torch.zeros(1))
            self.opacity_alpha = nn.Parameter(torch.zeros(1))
            self.shs_alpha = nn.Parameter(torch.zeros(1))
            # Independent alpha for each language feature
            self.lang_alphas = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_lang_features)])
            self.lang_alpha = nn.Parameter(torch.zeros(1))  # legacy for discrete mode
        else:
            self.register_buffer('pos_alpha', torch.ones(1))
            self.register_buffer('scales_alpha', torch.ones(1))
            self.register_buffer('rotations_alpha', torch.ones(1))
            self.register_buffer('opacity_alpha', torch.ones(1))
            self.register_buffer('shs_alpha', torch.ones(1))
            # Independent alpha for each language feature
            self.lang_alphas = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(self.num_lang_features)])
            self.register_buffer('lang_alpha', torch.ones(1))  # legacy for discrete mode
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1]) # rays_pts_emb[B,48] time_emb[B,1]
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            # import pdb; pdb.set_trace()
            hidden = torch.cat([grid_feature],-1) 
        
        # import pdb; pdb.set_trace()
        hidden = self.feature_out(hidden)   
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None,lang_emb=None, time_feature=None, time_sel_emb=None,time_pos_emb=None,init_centers=False):
        """
            time_pos_emb: (n,2*timebase_pe+1)
        """
        if time_sel_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, lang_emb, time_feature, time_sel_emb,time_pos_emb, init_centers)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, lang_emb, time_feature, time_sel_emb,time_pos_emb,init_centers=False):
        
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_sel_emb)

        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_alpha * self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_alpha * self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_alpha * self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_alpha * self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_alpha * self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb*mask.unsqueeze(-1) + dshs
        
        if os.getenv('use_discrete_lang_f','f') == 't' and init_centers == False:
            # Legacy discrete language feature mode (single head, full dim)
            lang_feature = lang_emb[:,:int(os.getenv("language_feature_hiddendim",3)*int(os.getenv("centers_num",3)))]
            lang_feature = lang_feature.view(lang_feature.shape[0], int(os.getenv("centers_num",3)), -1)
            lang_feature = lang_feature / (torch.norm(lang_feature, dim=-1, keepdim=True))
            coff = self.discrete_coff_generator(hidden)
            
            lang_feature = torch.matmul(coff.unsqueeze(1), lang_feature).squeeze(1)
            lang_feature = lang_feature / (torch.norm(lang_feature, dim=1, keepdim=True) + + 1e-9) 
        else:
            coff = None
            assert (init_centers and self.args.no_dlang)==False , " Dlang must be enabled when initialized centers"
 
            if self.args.no_dlang:
                # No deformation: just pass through (but still normalize each feature independently)
                lang_features_out = []
                for i in range(self.num_lang_features):
                    start_idx = i * self.lang_feature_dim
                    end_idx = start_idx + self.lang_feature_dim
                    feat = lang_emb[:, start_idx:end_idx]
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-9)
                    lang_features_out.append(feat)
                lang_feature = torch.cat(lang_features_out, dim=-1)
            else:
                # Process each language feature independently through its own head
                lang_features_out = []
                for i in range(self.num_lang_features):
                    start_idx = i * self.lang_feature_dim
                    end_idx = start_idx + self.lang_feature_dim
                    feat_in = lang_emb[:, start_idx:end_idx]
                    
                    if os.getenv("use_tribute_dlang","f") == "t":
                        # Use hidden features only (no lang input to MLP)
                        dlang_i = self.lang_alphas[i] * self.lang_deforms[i](
                            torch.cat([torch.zeros_like(feat_in), time_pos_emb], dim=1)
                        )
                    else:
                        # Use feature + time positional encoding as input
                        dlang_i = self.lang_alphas[i] * self.lang_deforms[i](
                            torch.cat([feat_in, time_pos_emb], dim=1)
                        )
                    
                    if os.getenv("no_resnet",'f') == 't':
                        feat_out = dlang_i
                    else:
                        feat_out = feat_in * mask + dlang_i
                    
                    # Normalize this feature independently
                    feat_out = feat_out / (feat_out.norm(dim=-1, keepdim=True) + 1e-9)
                    lang_features_out.append(feat_out)
                
                # Concatenate all features for output (for rasterizer compatibility)
                lang_feature = torch.cat(lang_features_out, dim=-1)

        return pts, scales, rotations, opacity, shs, lang_feature, coff
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, lang=None,times_sel=None,init_centers=False):
        return self.forward_dynamic(point, scales, rotations, opacity, shs,lang, times_sel,init_centers)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None,lang=None, times_sel=None,init_centers=False):
        times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)

        means3D, scales, rotations, opacity, shs, lang, coff  = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                lang,
                                                None,
                                                times_sel,
                                                times_emb,
                                                init_centers)
        return means3D, scales, rotations, opacity, shs, lang, coff
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb