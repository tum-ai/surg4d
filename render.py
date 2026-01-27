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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
from sklearn.decomposition import PCA
from scene.cameras import rotate_camera_around_center


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, os.path.join(path, "{0:05d}".format(count) + ".png")
            )
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)
to8b_np = lambda x: (255 * x).astype(np.uint8)
from typing import Literal


def pca_compress(rendering):
    feature_map = rendering.permute(1, 2, 0).cpu().numpy()
    # from PIL import Image
    pca = PCA(n_components=3)
    n = feature_map.shape[2]
    w = feature_map.shape[0]
    h = feature_map.shape[1]
    feature_map_reshaped = feature_map.reshape(-1, n)
    feature_map_pca = pca.fit_transform(feature_map_reshaped)
    feature_map_pca_reshaped = feature_map_pca.reshape(w, h, 3)
    # 将 feature map 归一化到 0-255
    feature_map_normalized = (
        feature_map_pca_reshaped - feature_map_pca_reshaped.min()
    ) / (feature_map_pca_reshaped.max() - feature_map_pca_reshaped.min())
    rendering = torch.from_numpy(feature_map_normalized)
    return rendering


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    cam_type,
    output_channel: Literal["rgb", "lang"],
    lf_path,
    data_type,
    args,
):
    ONLY_EVAL = os.getenv("ONLY_EVAL", "f")
    if ONLY_EVAL == "t":
        print("Only eval mode, no load ground truth feature.")
    if output_channel == "rgb":
        output_channel_key = "render"
    else:
        output_channel_key = "language_feature_image"
    save_name = f"{name}_{output_channel}"
    render_path = os.path.join(
        model_path, save_name, "ours_{}".format(iteration), "renders"
    )
    gts_path = os.path.join(model_path, save_name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(
        model_path, save_name, "ours_{}".format(iteration), "renders_npy"
    )
    gts_npy_path = os.path.join(
        model_path, save_name, "ours_{}".format(iteration), "gt_npy"
    )

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    render_images = []
    gt_list = []
    gt_nonorm_list = []
    render_list = []
    gt_images = []
    tosave_images_rendering = []
    print(f"name:{name}")
    print("point nums:", gaussians._xyz.shape[0])
    print(f"len:{len((views))}")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:
            time1 = time()

        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            None,
            cam_type=cam_type,
            args=args,
            stage=args.load_stage,
        )[output_channel_key]
        render_list.append(rendering)

        if output_channel == "rgb":
            gt = view.original_image[0:3, :, :]
            gt_nonorm_list.append(gt)

        else:
            if ONLY_EVAL == "t":
                gt, mask = None, None
                gt_nonorm_list.append(gt)
            else:
                gt, mask = view.get_language_feature(
                    language_feature_dir=lf_path,
                    feature_level=args.feature_level,
                    split=name,
                    data_type=data_type,
                )
                gt_nonorm_list.append(gt)
                if data_type != "dynerf" or name != "video":
                    gt = (gt + 1.0) / 2
            if rendering.shape[0] > 3:
                # Rendering and comparing instance features
                rendering = rendering[-3:, :, :]
                if ONLY_EVAL == "t":
                    gt = None
                else:
                    gt = gt[-3:, :, :]
        gt_list.append(gt)
        # Normalize rendering from [-1, 1] to [0, 1] for visualization (same as GT)
        if output_channel == "lang":
            rendering_viz = (rendering + 1.0) / 2
        else:
            rendering_viz = rendering
        tosave_images_rendering.append(rendering_viz)
        render_images.append(to8b(rendering_viz).transpose(1, 2, 0))

        if data_type != "dynerf" and name == "video":
            if ONLY_EVAL == "f":
                gt_images.append(to8b(gt).transpose(1, 2, 0))
            else:
                gt_images.append(None)

    time2 = time()
    print("FPS:", (len(views) - 1) / (time2 - time1))

    if not args.noimage:
        print("Saving images")
        if (data_type != "dynerf" or name != "video") and ONLY_EVAL == "f":
            multithread_write(gt_list, gts_path)
        multithread_write(tosave_images_rendering, render_path)
    else:
        print("For speed up, don't render image")

    if not args.nonpy:
        print("Saving npy")

        for idx in tqdm(range(len(gt_nonorm_list)), desc="Saving progress"):
            np.save(
                os.path.join(render_npy_path, "{0:05d}".format(idx) + ".npy"),
                render_list[idx].permute(1, 2, 0).cpu().numpy(),
            )
            if (data_type != "dynerf" or name != "video") and ONLY_EVAL == "f":
                np.save(
                    os.path.join(gts_npy_path, "{0:05d}".format(idx) + ".npy"),
                    gt_nonorm_list[idx].permute(1, 2, 0).cpu().numpy(),
                )
    else:
        print("For speed up, don't render npy")

    if not args.novideo:
        print("Saving video")
        imageio.mimwrite(
            os.path.join(
                model_path,
                save_name,
                "ours_{}".format(iteration),
                f"video_{output_channel}.mp4",
            ),
            render_images,
            fps=30,
        )
        if (data_type != "dynerf" and name == "video") and ONLY_EVAL == "f":
            imageio.mimwrite(
                os.path.join(
                    model_path,
                    save_name,
                    "ours_{}".format(iteration),
                    f"video_gt_{output_channel}.mp4",
                ),
                gt_images,
                fps=30,
            )
    else:
        print("For speed up, don't render video")


def render_sets(
    dataset: ModelParams,
    hyperparam,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    skip_video: bool,
    mode: Literal["rgb", "lang"],
    args,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            load_stage=args.load_stage,
        )
        cam_type = scene.dataset_type
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                mode,
                dataset.lf_path,
                scene.dataset_type,
                args,
            )
        if not skip_video:
            render_set(
                dataset.model_path,
                "video",
                scene.loaded_iter,
                scene.getVideoCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                mode,
                dataset.lf_path,
                scene.dataset_type,
                args,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                mode,
                dataset.lf_path,
                scene.dataset_type,
                args,
            )

        # if not args.skip_new_view:
        #     render_new_view_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background,cam_type,mode,dataset.lf_path,scene.dataset_type,args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    # parser.add_argument("--skip_new_view", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--mode", choices=["rgb", "lang"], default="rgb")
    parser.add_argument("--novideo", type=int, default=0)
    parser.add_argument("--noimage", type=int, default=0)
    parser.add_argument("--nonpy", type=int, default=0)
    parser.add_argument("--load_stage", type=str, default="fine-lang")
    args = parser.parse_args()
    # args = get_combined_args(parser)
    print("Rendering ", args.model_path)
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
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
        args.mode,
        args,
    )
