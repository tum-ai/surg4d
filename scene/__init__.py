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
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
class Scene:

    gaussians : GaussianModel

    @staticmethod
    def _best_stage_dirname(stage: str) -> str:
        return f"{stage}_best"

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False,load_stage='fine-lang'):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        self.load_best = False
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"),load_stage)
            elif load_iteration == -2:
                self.load_best = True
            else:
                self.loaded_iter = load_iteration
            if self.load_best:
                print(f"Loading best trained model for stage {load_stage}")
            else:
                print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        # import pdb; pdb.set_trace()
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
            dataset_type="blender"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type="PanopticSports"
        elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
            scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
            dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        
        # Load CoTracker data if available
        from utils.cotracker_gaussian_utils import load_cotracker_data
        from pathlib import Path
        cotracker_data = load_cotracker_data(Path(args.source_path))
        if cotracker_data is not None:
            self.gaussians._cotracker_data = cotracker_data
            # Store number of frames for time-to-frame conversion
            self.gaussians._num_frames = cotracker_data["gaussian_positions_precomputed"].shape[0]
        
        if self.loaded_iter or self.load_best:
            if self.load_best:
                point_cloud_dir = os.path.join(
                    self.model_path,
                    "point_cloud",
                    self._best_stage_dirname(load_stage),
                )
            else:
                point_cloud_dir = os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"{load_stage}_iteration_" + str(self.loaded_iter),
                )
            self.gaussians.load_ply(os.path.join(point_cloud_dir, "point_cloud.ply"))
            self.gaussians.load_model(point_cloud_dir)
            # Initialize CoTracker control-point-driven mask if data is available (same as create_from_pcd)
            if cotracker_data is not None:
                from utils.cotracker_gaussian_utils import initialize_control_point_driven_mask
                n_gaussians = self.gaussians.get_xyz.shape[0]
                gaussian_control_point_indices = cotracker_data["gaussian_control_point_indices"]
                self.gaussians._is_control_point_driven = initialize_control_point_driven_mask(
                    n_gaussians, gaussian_control_point_indices
                ).cuda()
                
                # Load precomputed positions (already torch tensor)
                gaussian_positions_precomputed = cotracker_data["gaussian_positions_precomputed"]
                self.gaussians._control_point_positions_precomputed = gaussian_positions_precomputed.float().cuda()  # (T, N_gaussians, 3)

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage, best=False):
        if best:
            point_cloud_path = os.path.join(
                self.model_path,
                "point_cloud",
                self._best_stage_dirname(stage),
            )
        else:
            point_cloud_path = os.path.join(self.model_path, f"point_cloud/{stage}_iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    def getVideoCameras(self, scale=1.0):
        return self.video_camera