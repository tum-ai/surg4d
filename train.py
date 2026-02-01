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
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, cos_loss
from utils.flow_utils import compute_raft_flow, compute_gaussian_flow, compute_flow_loss, visualize_flow
from gaussian_renderer import render, network_gui, render_opacity
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.vis_utils import (
    get_camera_intrinsics_from_fov,
    get_c2w_from_camera,
    sample_points_with_rgb
)
import rerun as rr
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from PIL import Image
from typing import Literal
from loguru import logger
import wandb

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_progressive_window_indices(
    iteration: int,
    total_frames: int,
    center_frame_idx: int,
    warmup_iterations: int,
) -> set[int]:
    """Compute frame indices for the current progressive training window.
    
    The window starts centered on center_frame_idx with size 1 and linearly
    expands to cover all frames over warmup_iterations.
    
    Args:
        iteration: Current training iteration (1-indexed in training loop).
        total_frames: Total number of frames in the dataset.
        center_frame_idx: Frame index to center the window on.
        warmup_iterations: Number of iterations until window covers all frames.
    
    Returns:
        Set of valid frame indices for the current window.
    """
    assert warmup_iterations > 0, "Warmup iterations must be greater than 0"

    # Compute progress ratio (0 to 1)
    progress = min(iteration / warmup_iterations, 1.0)
    
    # Window size grows from 1 to total_frames
    window_size = max(1, int(1 + (total_frames - 1) * progress))
    
    # Compute window bounds centered on center_frame_idx
    half_window = window_size // 2
    start_idx = max(0, center_frame_idx - half_window)
    end_idx = min(total_frames, start_idx + window_size)
    
    return set(range(start_idx, end_idx))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def concat_images(images, mode="vertical"):
    widths, heights = zip(*(i.size for i in images))

    if mode == "vertical":
        total_width = max(widths)
        total_height = sum(heights)
    else:  # horizontal
        total_width = sum(widths)
        total_height = max(heights)

    new_im = Image.new("RGB", (total_width, total_height))

    y_offset = 0
    x_offset = 0
    for im in images:
        if mode == "vertical":
            new_im.paste(im, (0, y_offset))
            y_offset += im.height
        else:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.width

    return new_im


def image2save(image, mode):
    if mode == "lang":
        image = (image + 1.0) / 2
        if image.shape[0] > 3:
            feature_map = image.permute(1, 2, 0).detach().cpu().numpy()
            from sklearn.decomposition import PCA

            # from PIL import Image
            pca = PCA(n_components=3)
            n = feature_map.shape[2]
            w = feature_map.shape[0]
            h = feature_map.shape[1]
            feature_map_reshaped = feature_map.reshape(-1, n)
            feature_map_pca = pca.fit_transform(feature_map_reshaped)
            feature_map_pca_reshaped = feature_map_pca.reshape(w, h, 3)
            # Normalize the feature map to the range [0, 1]
            feature_map_normalized = (
                feature_map_pca_reshaped - feature_map_pca_reshaped.min()
            ) / (feature_map_pca_reshaped.max() - feature_map_pca_reshaped.min())
            image_np = feature_map_normalized
            return Image.fromarray((image_np * 255).astype(np.uint8))

    image_np = image.detach().permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1)
    image_output = Image.fromarray((image_np * 255).astype(np.uint8))
    return image_output


def scene_reconstruction(
    dataset,
    opt,
    hyper,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    gaussians,
    scene,
    stage: Literal[
        "coarse-base", "coarse-lang", "fine-base", "fine-lang", "fine-lang-discrete"
    ],
    joint_train,
    tb_writer,
    train_iter,
    args,
    timer,
):
    first_iter = 0

    if joint_train == True:
        assert "lang" in stage
    logger.info(
        f"stage:{stage} begin... train_iter:{train_iter}, joint_train:{joint_train}"
    )
    
    # Log progressive window settings for fine stages
    if "fine" in stage and hasattr(args, "progressive_window_enabled") and args.progressive_window_enabled:
        center_idx = getattr(args, "progressive_window_center_frame_idx", None)
        if center_idx is None:
            center_idx = getattr(args, "coarse_frame_idx", 0) or 0
        logger.info(
            f"Progressive window enabled: warmup_iters={args.progressive_window_warmup_iterations}, "
            f"center_frame={center_idx}"
        )
    
    if "discrete" in stage:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(
            model_params,
            opt,
            stage=stage,
            joint_train=joint_train,
            no_dlang=args.no_dlang,
            init_from_stage=args.init_from_stage,
            coarse_freeze_xyz=args.coarse_freeze_xyz,
        )
    else:
        gaussians.training_setup(opt, stage, joint_train, args.no_dlang, coarse_freeze_xyz=args.coarse_freeze_xyz)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    # Initialize rerun logging for fine stages
    rerun_initialized = False
    if "fine" in stage:
        rerun_save_path = os.path.join(scene.model_path, f"rerun_{stage}.rrd")
        rr.init(f"training_{stage}", spawn=False)
        rr.save(rerun_save_path)
        rr.log("/", rr.ViewCoordinates.RDF, static=True)  # Right-Down-Forward (OpenCV convention)
        rerun_initialized = True
        logger.info(f"Rerun logging initialized, will save to {rerun_save_path}")

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size if "base" in stage else 1
    logger.info("data loading done")

    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                collate_fn=list,
                worker_init_fn=seed_worker,
            )
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=list,
                worker_init_fn=seed_worker,
            )
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    load_in_memory = False
    #
    count = 0

    log_iter_interval = 500

    save_path = os.path.join(scene.model_path, "training_output_img")
    os.makedirs(save_path, exist_ok=True)
    total_time_ongt = 0

    # Consistent indices for logging
    num_gaussians = gaussians.get_xyz.shape[0]
    print(f"#gaussians: {num_gaussians}")
    sample_indices = torch.randperm(num_gaussians, device=gaussians.get_xyz.device)[:num_gaussians // 4]
    
    for iteration in range(first_iter, final_iter + 1):
        # seed_everything(6666)
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    count += 1
                    viewpoint_index = (count) % len(video_cams)
                    if (count // (len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1

                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time

                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifer,
                        stage=stage,
                        cam_type=scene.dataset_type,
                        args=args,
                    )["render"]

                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # Check if we should use a single frame for coarse training
        use_single_frame = (
            "coarse" in stage
            and hasattr(args, "coarse_frame_idx")
            and args.coarse_frame_idx is not None
        )
        
        # Check if we should use progressive window for fine stages
        use_progressive_window = (
            "fine" in stage
            and hasattr(args, "progressive_window_enabled")
            and args.progressive_window_enabled
        )

        if use_single_frame:
            # Overfit on a single frame during coarse stages
            viewpoint_cams = [temp_list[args.coarse_frame_idx]]
        # dynerf's branch
        elif opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                logger.info("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(
                        viewpoint_stack,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=32,
                        collate_fn=list,
                    )
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            # Determine which frames are available for sampling
            if use_progressive_window:
                # Get valid frame indices for current iteration
                valid_indices = get_progressive_window_indices(
                    iteration=iteration,
                    total_frames=len(temp_list),
                    center_frame_idx=args.progressive_window_center_frame_idx,
                    warmup_iterations=args.progressive_window_warmup_iterations,
                )
                # Filter to frames in the window
                available_frames = [temp_list[i] for i in valid_indices]
            else:
                available_frames = temp_list
            
            # Refill viewpoint_stack from available frames if empty
            if not viewpoint_stack:
                viewpoint_stack = available_frames.copy()
            elif use_progressive_window:
                # Add newly available frames immediately when window expands
                # Use uid to match cameras (can't use object identity due to deepcopy)
                uids_in_stack = {cam.uid for cam in viewpoint_stack}
                new_frames = [temp_list[i] for i in valid_indices if temp_list[i].uid not in uids_in_stack]
                viewpoint_stack.extend(new_frames)
            
            idx = 0
            viewpoint_cams = []

            while idx < batch_size:
                viewpoint_cam = viewpoint_stack.pop(
                    randint(0, len(viewpoint_stack) - 1)
                )
                if not viewpoint_stack:
                    viewpoint_stack = available_frames.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))
        # breakpoint()
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        gt_language_features = []
        language_features = []
        language_feature_masks = []
        coff_list = []
        depth_maps = []
        gt_depth_maps = []

        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                opt,
                stage=stage,
                cam_type=scene.dataset_type,
                args=args,
            )
            
            # Get gt_image for current frame
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam["image"].cuda()
            
            (
                image,
                language_feature,
                viewspace_point_tensor,
                visibility_filter,
                radii,
            ) = (
                render_pkg["render"],
                render_pkg["language_feature_image"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            coff = render_pkg["coff"]
            if coff is not None:
                coff_list.append(coff)

            images.append(image.unsqueeze(0))
            start_time = time.time()
            gt_images.append(gt_image.unsqueeze(0))

            depth_map = render_pkg["depth"]
            depth_maps.append(depth_map.unsqueeze(0))
            gt_depth_map = viewpoint_cam.get_depth_map(
                dataset.depth_path, split="train", data_type=scene.dataset_type
            )
            gt_depth_maps.append(gt_depth_map.unsqueeze(0))

            if "base" not in stage:
                gt_language_feature, language_feature_mask = (
                    viewpoint_cam.get_language_feature(
                        language_feature_dir=dataset.lf_path,
                        feature_level=dataset.feature_level,
                        data_type=scene.dataset_type,
                    )
                )
                gt_language_features.append(gt_language_feature)
            end_time = time.time()
            total_time_ongt += end_time - start_time

            if "base" not in stage:
                language_feature_masks.append(language_feature_mask)
                language_features.append(language_feature)

            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        if "base" not in stage:
            language_feature_tensor = torch.cat(language_features, 0)
            language_feature_mask_tensor = torch.cat(language_feature_masks, 0)
            gt_language_feature_tensor = torch.cat(gt_language_features, 0)
        depth_map_tensor = torch.cat(depth_maps, 0)  # (batch, 1, h, w)
        gt_depth_map_tensor = torch.cat(gt_depth_maps, 0)

        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        # Loss
        # breakpoint()
        # print(dataset.lf_path)
        resdict = {}
        # Image-based stage
        if "base" in stage:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
            resdict["rgb_l1"] = Ll1.item()

            if args.depth_loss_weight != 0.0:
                depth_loss = l1_loss(depth_map_tensor, gt_depth_map_tensor)
                resdict["depth_l1"] = depth_loss.item()
                Ll1 += args.depth_loss_weight * depth_loss
        # Language feature stage
        else:
            Ll1 = args.lam * l1_loss(
                language_feature_tensor * language_feature_mask_tensor,
                gt_language_feature_tensor * language_feature_mask_tensor,
            )
            resdict["lang_l1"] = Ll1.item()

            # if os.getenv("addcosloss", "f") == "t":
            # # TODO: compare L1 and cos and make sure to put in configs
            # Ll1 = 0.0
            # # TODO: experiment with what this masking does?
            # language_feature_mask_tensor = torch.ones_like(language_feature_mask_tensor)
            # cosloss = cos_loss(
            #     language_feature_tensor * language_feature_mask_tensor,
            #     gt_language_feature_tensor * language_feature_mask_tensor,
            # )
            # print(f"weighted cosloss: {args.beta * cosloss.item()} with beta {args.beta}")
            # Ll1 += args.beta * cosloss
            # resdict["lang_l1"] = cosloss.item()

            if joint_train:
                Ll1_rgb = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
                resdict["rgb_l1"] = Ll1_rgb.item()
                Ll1 += Ll1_rgb
        
        # if opt.include_feature:
        #     Ll1 += l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)
        # loss = Ll1

        # psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm

        loss = Ll1
        if os.getenv("wandb", "f") == "t":
            wandb.log(resdict)

        ################ DEBUG  ################
        if iteration % log_iter_interval == 0:
            if "base" in stage:
                images = [
                    image2save(image, "rgb"),
                    image2save(gt_image, "rgb"),
                ]
            else:
                if language_feature is None:
                    language_feature = torch.zeros_like(gt_language_feature)
                images = [
                    image2save(image, "rgb"),
                    image2save(gt_image, "rgb"),
                    image2save(language_feature, "lang"),
                    image2save(gt_language_feature, "lang"),
                ]
            concatenated_image = concat_images(images, mode="horizontal")

            #
            concatenated_image.save(
                os.path.join(save_path, f"output_{stage}_{iteration}_{viewpoint_cam.colmap_id}.png")
            )
            if os.getenv("wandb", "f") == "t":
                wandb.log({"training images": wandb.Image(concatenated_image)})
            
            # Rerun logging for fine stages - log camera poses and point clouds
            if rerun_initialized and "fine" in stage:
                
                for cam_idx, viewpoint_cam in enumerate(viewpoint_cams):
                    # Get camera intrinsics and extrinsics
                    K = get_camera_intrinsics_from_fov(
                        viewpoint_cam.FoVx, viewpoint_cam.FoVy,
                        viewpoint_cam.image_width, viewpoint_cam.image_height
                    )
                    c2w = get_c2w_from_camera(viewpoint_cam)                    

                    rr.set_time_sequence("timestep", sequence=int(viewpoint_cam.time * 1000))
                    
                    # Log camera transform (camera-to-world)
                    rr.log(
                        f"world/camera_{cam_idx}",
                        rr.Transform3D(
                            translation=c2w[:3, 3].detach().cpu().numpy(),
                            mat3x3=c2w[:3, :3].detach().cpu().numpy(),
                        ),
                    )
                    
                    # Log camera pinhole model
                    rr.log(
                        f"world/camera_{cam_idx}",
                        rr.Pinhole(
                            resolution=[viewpoint_cam.image_width, viewpoint_cam.image_height],
                            focal_length=[K[0, 0].item(), K[1, 1].item()],
                            principal_point=[K[0, 2].item(), K[1, 2].item()],
                            camera_xyz=rr.ViewCoordinates.RDF,
                            image_plane_distance=0.1,
                        ),
                    )
                    
                    # Log the ground truth image
                    if scene.dataset_type != "PanopticSports":
                        gt_img_np = viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy()
                    else:
                        gt_img_np = viewpoint_cam["image"].permute(1, 2, 0).cpu().numpy()
                    gt_img_np = (gt_img_np * 255).astype(np.uint8)
                    rr.log(f"world/camera_{cam_idx}/image", rr.Image(gt_img_np))
                    
                    # Generate and log point cloud from depth map (25% sampled)
                    gt_depth = viewpoint_cam.get_depth_map(
                        dataset.depth_path, split="train", data_type=scene.dataset_type
                    )
                    if gt_depth is not None:
                        gt_depth_squeezed = gt_depth.squeeze(0).cuda()  # (H, W)
                        if scene.dataset_type != "PanopticSports":
                            gt_rgb_for_pc = viewpoint_cam.original_image.cuda()
                        else:
                            gt_rgb_for_pc = viewpoint_cam["image"].cuda()
                        
                        # # Sample 25% of points
                        # points, colors = sample_points_with_rgb(
                        #     gt_depth_squeezed, gt_rgb_for_pc, K.cuda(), c2w.cuda(),
                        #     sample_ratio=0.25
                        # )
                        
                        # if len(points) > 0:
                        #     # Convert colors from [0,1] to [0,255] uint8
                        #     colors_uint8 = (colors.detach().cpu().numpy() * 255).astype(np.uint8)
                        #     rr.log(
                        #         "world/point_cloud_depth",
                        #         rr.Points3D(
                        #             positions=points.detach().cpu().numpy(),
                        #             colors=colors_uint8,
                        #             radii=0.002,
                        #         ),
                        #     )
                
                    # Log BOTH base and deformed Gaussian means for comparison
                    # Get base Gaussian properties - use raw parameters like the renderer does
                    means3D = gaussians.get_xyz
                    scales = gaussians._scaling  # Raw, not get_scaling (which applies activation)
                    rotations = gaussians._rotation  # Raw, not get_rotation
                    opacity = gaussians._opacity  # Raw
                    shs = gaussians.get_features
                    language_feature = gaussians.get_language_feature
                    
                    # Get colors from base SH DC component (shared)
                    sampled_shs = shs[sample_indices, 0, :].detach().cpu().numpy()
                    sampled_colors_rgb = (sampled_shs * 0.28209479177387814 + 0.5).clip(0, 1)
                    sampled_colors_uint8 = (sampled_colors_rgb * 255).clip(0, 255).astype(np.uint8)
                    
                    # # Log BASE (undeformed) Gaussian positions in BLUE
                    # sampled_base_xyz = means3D[sample_indices].detach().cpu().numpy()
                    # rr.log(
                    #     "world/gaussians_base",
                    #     rr.Points3D(
                    #         positions=sampled_base_xyz,
                    #         colors=np.full((len(sampled_base_xyz), 3), [0, 100, 255], dtype=np.uint8),  # Blue
                    #         radii=0.002,
                    #     ),
                    # )
                    
                    # Create time tensor for all Gaussians - use repeat() like the renderer
                    time_tensor = torch.tensor(viewpoint_cam.time).to(means3D.device).repeat(means3D.shape[0], 1)
                    
                    # Apply deformation to get deformed positions
                    with torch.no_grad():
                        means3D_deformed, scales_deformed, rotations_deformed, opacity_deformed, shs_deformed, lang_deformed, coff = gaussians._deformation(
                            means3D, scales, rotations, opacity, shs, language_feature, time_tensor
                        )
                    
                    # # Log DEFORMED Gaussian positions with actual colors
                    # sampled_deformed_xyz = means3D_deformed[sample_indices].detach().cpu().numpy()
                    # rr.log(
                    #     "world/gaussians_deformed",
                    #     rr.Points3D(
                    #         positions=sampled_deformed_xyz,
                    #         colors=sampled_colors_uint8,
                    #         radii=0.003,
                    #     ),
                    # )

                    # Log the precomputed positions, using viewpoint_cam.time to get the frame index
                    if gaussians._control_point_positions_precomputed is not None and gaussians._is_control_point_driven is not None:
                        frame_idx = int(time_tensor[0, 0].item() * (gaussians._num_frames - 1))
                        precomputed_positions = gaussians._control_point_positions_precomputed[frame_idx]
                        
                        if len(sample_indices) > 0:
                            sampled_precomputed_positions = precomputed_positions[sample_indices].detach().cpu().numpy()
                            rr.log(
                                "world/gaussians_precomputed",
                                rr.Points3D(
                                    positions=sampled_precomputed_positions,
                                    # Use the colors of the control point driven Gaussians
                                    colors=sampled_colors_uint8,
                                    radii=0.003,
                                ),
                            )

                    
        # print(f"language_feature.max():{language_feature.max()},language_feature.min():f{language_feature.min()}")
        ################################

        if "fine" in stage and hyper.time_smoothness_weight != 0:
            # plane tv is smoothness of spatial dimensions
            # time smoothness is smoothness of temporal dimensions
            # l1 time planes is sparsity of temporal dimensions (encourages constant values)
            tv_loss = gaussians.compute_regulation(
                hyper.time_smoothness_weight,
                hyper.l1_time_planes,
                hyper.plane_tv_weight,
            )
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0 - ssim_loss)

        loss.backward()

        if torch.isnan(loss).any():
            logger.info("loss", loss)
            logger.info("loss is nan,end training, reexecv program now.")

            os.execv(sys.executable, [sys.executable] + sys.argv)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = (
                viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            )
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        #   "psnr": f"{psnr_:.{2}f}",
                        "point": f"{total_point}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                [pipe, background, opt],
                stage,
                scene.dataset_type,
                args,
            )
            if (iteration in saving_iterations) and "coarse-base" not in stage:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (
                    (iteration < 1000 and iteration % 10 == 9)
                    or (iteration < 3000 and iteration % 50 == 49)
                    or (iteration < 60000 and iteration % 100 == 99)
                ):
                    if len(test_cams) > 0:
                        render_training_image(
                            scene,
                            gaussians,
                            [test_cams[iteration % len(test_cams)]],
                            render,
                            pipe,
                            background,
                            opt,
                            stage + "test",
                            iteration,
                            timer.get_elapsed_time(),
                            scene.dataset_type,
                            args,
                        )
                    render_training_image(
                        scene,
                        gaussians,
                        [train_cams[iteration % len(train_cams)]],
                        render,
                        pipe,
                        background,
                        opt,
                        stage + "train",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                        args,
                    )

            timer.start()

            if iteration < opt.densify_until_iter and "base" in stage:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter
                )

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
                        opt.opacity_threshold_fine_init
                        - opt.opacity_threshold_fine_after
                    ) / (opt.densify_until_iter)
                    densify_threshold = (
                        opt.densify_grad_threshold_fine_init
                        - iteration
                        * (
                            opt.densify_grad_threshold_fine_init
                            - opt.densify_grad_threshold_after
                        )
                        / (opt.densify_until_iter)
                    )
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    gaussians.densify(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        5,
                        5,
                        scene.model_path,
                        iteration,
                        stage,
                    )
                if (
                    iteration > opt.pruning_from_iter
                    and iteration % opt.pruning_interval == 0
                    and gaussians.get_xyz.shape[0] > 200000
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    gaussians.prune(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        stage,
                    )

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if (
                    iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                    and opt.add_point
                ):
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    logger.info("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations) and "fine" in stage:
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(opt.include_feature), iteration),
                    scene.model_path
                    + "/chkpnt"
                    + f"_{stage}_"
                    + str(iteration)
                    + ".pth",
                )
    logger.info(f"total_time_ongt:{total_time_ongt}")


def training(
    dataset,
    hyper,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    expname,
    timestamp,
    args,
):
    # first_iter = 0
    opt.iterations = (
        opt.coarse_base_iterations
        + opt.coarse_lang_iterations
        + opt.fine_base_iterations
        + opt.fine_lang_iterations
    )
    tb_writer = prepare_output_and_logger(os.path.join(expname), timestamp)
    logger.info(f"Model Path:{args.model_path}")
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    if args.resume_from_final_stage == 1:
        timer.start()
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=args.resume_from_final_stage_load_iter,
            load_stage=args.init_from_stage,
        )
        scene_reconstruction(
            dataset,
            opt,
            hyper,
            pipe,
            testing_iterations,
            saving_iterations,
            checkpoint_iterations,
            checkpoint,
            debug_from,
            gaussians,
            scene,
            "fine-lang-discrete",
            args.joint_fine,
            tb_writer,
            opt.fine_lang_iterations + 10000,
            args,
            timer,
        )
    else:
        # Allow resuming from a specific saved stage/iteration (non-discrete path)
        if getattr(args, "resume_from_stage", ""):
            timer.start()
            scene = Scene(
                dataset,
                gaussians,
                load_iteration=getattr(args, "resume_from_iter", -1),
                load_stage=args.resume_from_stage,
            )
        else:
            scene = Scene(dataset, gaussians, load_coarse=None)
        logger.info(f"opt.coarse_base_iterations:{opt.coarse_base_iterations}")
        logger.info(f"opt.coarse_lang_iterations:{opt.coarse_lang_iterations}")
        logger.info(f"opt.fine_base_iterations:{opt.fine_base_iterations}")
        logger.info(f"opt.fine_lang_iterations:{opt.fine_lang_iterations}")
        logger.info(f"opt.iterations:{opt.iterations}")
        timer.start()

        if opt.coarse_base_iterations > 0:
            scene_reconstruction(
                dataset,
                opt,
                hyper,
                pipe,
                testing_iterations,
                saving_iterations,
                checkpoint_iterations,
                checkpoint,
                debug_from,
                gaussians,
                scene,
                "coarse-base",
                False,
                tb_writer,
                opt.coarse_base_iterations,
                args,
                timer,
            )
        if opt.coarse_lang_iterations > 0:
            scene_reconstruction(
                dataset,
                opt,
                hyper,
                pipe,
                testing_iterations,
                saving_iterations,
                checkpoint_iterations,
                checkpoint,
                debug_from,
                gaussians,
                scene,
                "coarse-lang",
                args.joint_coarse,
                tb_writer,
                opt.coarse_lang_iterations,
                args,
                timer,
            )
        if opt.fine_base_iterations > 0:
            scene_reconstruction(
                dataset,
                opt,
                hyper,
                pipe,
                testing_iterations,
                saving_iterations,
                checkpoint_iterations,
                checkpoint,
                debug_from,
                gaussians,
                scene,
                "fine-base",
                False,
                tb_writer,
                opt.fine_base_iterations,
                args,
                timer,
            )
        if opt.fine_lang_iterations > 0:
            scene_reconstruction(
                dataset,
                opt,
                hyper,
                pipe,
                testing_iterations,
                saving_iterations,
                checkpoint_iterations,
                checkpoint,
                debug_from,
                gaussians,
                scene,
                "fine-lang",
                args.joint_fine,
                tb_writer,
                opt.fine_lang_iterations,
                args,
                timer,
            )


def prepare_output_and_logger(expname, timestamp=None):
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join(os.getenv("ExpsDir", "./output"), unique_str)
    # Set up output folder
    logger.info("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer with timestamped log directory to avoid run name conflicts
    tb_writer = None
    if TENSORBOARD_FOUND:
        if timestamp is None:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Create a subdirectory with timestamp for TensorBoard logs
        tb_log_dir = os.path.join(args.model_path, "tensorboard_logs", timestamp)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_log_dir)
        logger.info("TensorBoard log directory: {}".format(tb_log_dir))
    else:
        logger.info("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    stage,
    dataset_type,
    args,
):
    if tb_writer:
        tb_writer.add_scalar(
            f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration
        )
        tb_writer.add_scalar(
            f"{stage}/train_loss_patches/total_loss", loss.item(), iteration
        )
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #
        validation_configs = (
            {
                "name": "test",
                "cameras": (
                    []
                    if len(scene.getTestCameras()) == 0
                    else [
                        scene.getTestCameras()[idx % len(scene.getTestCameras())]
                        for idx in range(10, 5000, 299)
                    ]
                ),
            },
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(10, 5000, 299)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            stage=stage,
                            cam_type=dataset_type,
                            args=args,
                            *renderArgs,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(
                            viewpoint.original_image.to("cuda"), 0.0, 1.0
                        )
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage
                                + "/"
                                + config["name"]
                                + "_view_{}/render".format(viewpoint.image_name),
                                image[None],
                                global_step=iteration,
                            )
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask

                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                logger.info(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if os.getenv("wandb", "f") == "t":
                    wandb.log({"psnr_test": psnr_test})
                # print("sh feature",scene.gaussians.get_features.shape)
                tb_writer.add_scalar(
                    stage + "/" + config["name"] + "/loss_viewpoint - l1_loss",
                    l1_test,
                    iteration,
                )
                tb_writer.add_scalar(
                    stage + "/" + config["name"] + "/loss_viewpoint - psnr",
                    psnr_test,
                    iteration,
                )

        if tb_writer:
            tb_writer.add_histogram(
                f"{stage}/scene/opacity_histogram",
                scene.gaussians.get_opacity,
                iteration,
            )

            tb_writer.add_scalar(
                f"{stage}/total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
            tb_writer.add_scalar(
                f"{stage}/deformation_rate",
                scene.gaussians._deformation_table.sum()
                / scene.gaussians.get_xyz.shape[0],
                iteration,
            )
            tb_writer.add_histogram(
                f"{stage}/scene/motion_histogram",
                scene.gaussians._deformation_accum.mean(dim=-1) / 100,
                iteration,
                max_bins=500,
            )

        torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    seed_everything(6666)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[2000, 10000, 20000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[2000, 10000, 20000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--joint_coarse", action="store_true")
    parser.add_argument("--joint_fine", action="store_true")
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--resume_from_final_stage", type=int, default=0)
    parser.add_argument("--resume_from_final_stage_load_iter", type=int, default=10000)
    parser.add_argument(
        "--init_from_stage", choices=["fine-lang", "fine-base"], default="fine-base"
    )
    parser.add_argument("--coff_time_smooth_loss_weight", type=float, default=1e-1)
    # Generic resume controls (non-discrete)
    parser.add_argument("--resume_from_stage", type=str, default="")
    parser.add_argument("--resume_from_iter", type=int, default=-1)
    # custom loss stuff
    parser.add_argument("--depth_loss_weight", type=float, default=0.0)

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        from utils.params_utils import merge_hparams

        config = None
        try:
            import mmcv  # type: ignore

            if hasattr(mmcv, "Config"):
                config = mmcv.Config.fromfile(args.configs)
        except Exception:
            config = None
        if config is None:
            try:
                from mmengine.config import Config as MMEngineConfig  # type: ignore

                config = MMEngineConfig.fromfile(args.configs)
            except Exception:
                raise ImportError(
                    "Neither mmcv.Config nor mmengine.config.Config is available; install mmcv<=1.x or mmengine."
                )
        args = merge_hparams(args, config)

    if os.getenv("wandb", "f") == "t":
        wandb.init(project="4DLangSplat", name=args.expname, config=args)

    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"log_{timestamp}_train.log"
    base_save_path = os.path.join(os.getenv("ExpsDir", "./output"), args.expname)
    logger.add(
        os.path.join(base_save_path, "log", log_file_name), rotation="500 MB"
    )  # 将日志写入文件，当文件大小达到500MB时进行轮转
    logger.info(args)
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    args.save_iterations.append(10000)
    args.save_iterations.append(20000)
    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
        timestamp,
        args,
    )

    # All done
    logger.info("\nTraining complete.")
