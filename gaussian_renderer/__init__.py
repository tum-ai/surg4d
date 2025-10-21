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
import time
import os
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt, scaling_modifier = 1.0, override_color = None, stage='fine-lang', cam_type=None,args=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    # if opt is not None:
    #     include_feature = opt.include_feature
    # else:
    #     include_feature = True
    if 'base' in stage:
        include_feature = False
    else:
        include_feature = True
    # include_feature = True
    # import pdb; pdb.set_trace()
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            include_feature=include_feature,
        )
        # time_smooth_frames = int(os.getenv("time_smooth_frames",0))
        
        # if viewpoint_camera.time > viewpoint_camera.time_interval*time_smooth_frames and viewpoint_camera.time < 1- viewpoint_camera.time_interval*time_smooth_frames:
        #     time_near = []
        #     for i in range(time_smooth_frames*2+1):
        #         time_near.append(torch.tensor(viewpoint_camera.time-viewpoint_camera.time_interval*time_smooth_frames+viewpoint_camera.time_interval*i).to(means3D.device).repeat(means3D.shape[0],1))
        # else:
        #     time_near = None

        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    if include_feature and 'base' not in stage:
        # import pdb; pdb.set_trace()
        language_feature_precomp = pc.get_language_feature
        if os.getenv("nonormalized",'f') == 'f':
            language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
        # language_feature_precomp = torch.sigmoid(language_feature_precomp)
    else:
        # language_feature_precomp = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
        #! 如果需要修改language feature需要修改这里
        language_feature_precomp = torch.zeros((pc._xyz.shape[0], int(os.getenv("language_feature_hiddendim",3))), dtype=opacity.dtype, device=opacity.device)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    coff = None
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, language_feature_precomp_final = means3D, scales, rotations, opacity, shs, language_feature_precomp
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        # import pdb; pdb.set_trace()
        if 'base' in stage:
            pc._deformation.deformation_net.args.no_dlang = 1
        else:
            pc._deformation.deformation_net.args.no_dlang = args.no_dlang

        means3D_final, scales_final, rotations_final, opacity_final, shs_final, language_feature_precomp_final, coff = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs, language_feature_precomp,
                                                                 time)
        # # import pdb; pdb.set_trace()
        # n = 100
        # language_feature_precomp_final = []
        # for _ in range(n):
        #     random_time = torch.rand_like(time)
        #     # print(random_time)
        #     language_feature_precomp_final.append(pc._deformation(means3D,scales,rotations,opacity,shs,language_feature_precomp,random_time)[-1])
        # res = torch.stack(language_feature_precomp_final,dim=1)
        # res = res.to('cpu')
        # from sklearn.cluster import KMeans
        # # 准备一个空列表来存储每组数据的中心点
        # centers = []

        # # 遍历每个 (10, 6) 的子张量
        # from tqdm import tqdm
        # for i in tqdm(range(res.shape[0])):
        #     # 获取当前的子张量
        #     data_point = res[i]  # 大小为 (10, 6)
            
        #     # 使用 KMeans 聚类，假设聚类成 1 个中心点（因为要得到一个中心点）
        #     kmeans = KMeans(n_clusters=3, random_state=0).fit(data_point.numpy())
            
        #     # 获取聚类中心点
        #     center = kmeans.cluster_centers_
            
        #     # 将中心点添加到列表
        #     centers.append(center)
            
        # centers_tensor = torch.tensor(centers).squeeze()  # 大小为 (136740, 6)
        # for id in tqdm(range(0,len(centers),1000)):
        #     # 将中心点列表转换为张量

        #     # print("聚类完成，中心点的形状为:", centers_tensor.shape)
        #     from sklearn.decomposition import PCA
        #     pca = PCA(n_components=2)
        #     centers = centers_tensor[id].numpy()
        #     datapoints = res[id].numpy()
        #     centers_2d = pca.fit_transform(centers)
        #     datapoints_2d = pca.transform(datapoints)
        #     import matplotlib.pyplot as plt
        #     # 绘制中心点和数据点
        #     plt.figure(figsize=(8, 6))
        #     plt.scatter(datapoints_2d[:, 0], datapoints_2d[:, 1], color='blue', label='Datapoints')
        #     plt.scatter(centers_2d[:, 0], centers_2d[:, 1], color='red', marker='X', s=100, label='Centers')

        #     plt.title('PCA Projection of Centers and Datapoints')
        #     plt.xlabel('PCA Component 1')
        #     plt.ylabel('PCA Component 2')
        #     plt.legend()
        #     plt.grid(True)
        #     os.makedirs('pca_output',exist_ok=True)
        #     plt.savefig(f'pca_output/result_{id}.png')
            
        # # import pdb; pdb.set_trace()# 使用 PCA 将数据降维到 2D

    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # import pdb; pdb.set_trace()
    # print(language_feature_precomp)
    # print("language_feature_precomp:",torch.isnan(language_feature_precomp).any())

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, language_feature_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        language_feature_precomp = language_feature_precomp_final,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # import pdb; pdb.set_trace()
    # print("rendered_image:",torch.isnan(rendered_image).any())
    # print("language_feature_image:",torch.isnan(language_feature_image).any())
    # print("radii:",torch.isnan(radii).any())
    # print("depth:",torch.isnan(depth).any())

    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if 'base' in stage:
        language_feature_image = None
    return {"render": rendered_image,
            "language_feature_image": language_feature_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "coff":coff}

def render_opacity(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt, scaling_modifier = 1.0, override_color = None, stage='fine-lang', cam_type=None,args=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    # if opt is not None:
    #     include_feature = opt.include_feature
    # else:
    #     include_feature = True
    if 'base' in stage:
        include_feature = False
    else:
        include_feature = True
    # include_feature = True
    # import pdb; pdb.set_trace()
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            include_feature=include_feature,
        )
        # time_smooth_frames = int(os.getenv("time_smooth_frames",0))
        
        # if viewpoint_camera.time > viewpoint_camera.time_interval*time_smooth_frames and viewpoint_camera.time < 1- viewpoint_camera.time_interval*time_smooth_frames:
        #     time_near = []
        #     for i in range(time_smooth_frames*2+1):
        #         time_near.append(torch.tensor(viewpoint_camera.time-viewpoint_camera.time_interval*time_smooth_frames+viewpoint_camera.time_interval*i).to(means3D.device).repeat(means3D.shape[0],1))
        # else:
        #     time_near = None

        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = torch.cat((
        pc._opacity.unsqueeze(-1).expand_as(pc._features_dc),
        torch.zeros_like(pc._features_rest)
    ), dim=1) # THIS CHANGE ENABLES OPACITY RENDERING


    if include_feature and 'base' not in stage:
        # import pdb; pdb.set_trace()
        language_feature_precomp = pc.get_language_feature
        if os.getenv("nonormalized",'f') == 'f':
            language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
        # language_feature_precomp = torch.sigmoid(language_feature_precomp)
    else:
        # language_feature_precomp = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
        #! 如果需要修改language feature需要修改这里
        language_feature_precomp = torch.zeros((pc._xyz.shape[0], int(os.getenv("language_feature_hiddendim",3))), dtype=opacity.dtype, device=opacity.device)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    coff = None
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, language_feature_precomp_final = means3D, scales, rotations, opacity, shs, language_feature_precomp
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        # import pdb; pdb.set_trace()
        if 'base' in stage:
            pc._deformation.deformation_net.args.no_dlang = 1
        else:
            pc._deformation.deformation_net.args.no_dlang = args.no_dlang

        means3D_final, scales_final, rotations_final, opacity_final, shs_final, language_feature_precomp_final, coff = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs, language_feature_precomp,
                                                                 time)


    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    
    # import pdb; pdb.set_trace()
    # print(language_feature_precomp)
    # print("language_feature_precomp:",torch.isnan(language_feature_precomp).any())

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, language_feature_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        language_feature_precomp = language_feature_precomp_final,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # import pdb; pdb.set_trace()
    # print("rendered_image:",torch.isnan(rendered_image).any())
    # print("language_feature_image:",torch.isnan(language_feature_image).any())
    # print("radii:",torch.isnan(radii).any())
    # print("depth:",torch.isnan(depth).any())

    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return rendered_image