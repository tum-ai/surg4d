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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import os
from sklearn.decomposition import PCA

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, depth=None, cam_name=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        # import pdb; pdb.set_trace()
        self.time = time
        self.cam_name = cam_name
        # print(f"cam_name:{cam_name}")
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.original_image = image.clamp(0.0, 1.0)[:3,:,:]

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask

        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
  
        self.depth = depth
        self.mask = mask
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # .cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_language_feature(self, language_feature_dir, feature_level,split='train',data_type="nerfies"):
        if data_type == "nerfies":
            if split=='train':
                real_id = self.colmap_id*4 + 1
            elif split == 'test': # test
                real_id = self.colmap_id*4 + 3
            else: # video
                real_id = self.colmap_id +1
            language_feature_name = os.path.join(language_feature_dir, f"{real_id:06}")

        elif data_type == "dynerf": 
            frame_id = self.colmap_id % 300

            if split ==  'test':
                assert self.colmap_id<300
            elif split == "video":
                return None, None

            language_feature_name = os.path.join(language_feature_dir, f"{self.cam_name}-{frame_id:04}")

        elif data_type == "colmap":
            # TODO: this is hardcoded, need to fix this! has to do with how colmap generated more images that are blurred
            #  and essentially repeat the dataset every few times
            # frame_id = self.colmap_id % 80
            frame_id = self.colmap_id + 1
            # if frame_id == 0:
            #     frame_id = 80
            language_feature_name = os.path.join(language_feature_dir, f"{frame_id:06}")            
        else:
            raise NotImplementedError

        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))


        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[:, y, x].squeeze(-1).long()
        mask = seg != -1
        if feature_level == 0: # default
            point_feature1 = feature_map[seg[0:1]].squeeze(0)
            mask = mask[0:1].reshape(1, self.image_height, self.image_width)
        elif feature_level == 1: # s
            point_feature1 = feature_map[seg[1:2]].squeeze(0)
            mask = mask[1:2].reshape(1, self.image_height, self.image_width)
        elif feature_level == 2: # m
            point_feature1 = feature_map[seg[2:3]].squeeze(0)
            mask = mask[2:3].reshape(1, self.image_height, self.image_width)
        elif feature_level == 3: # l
            point_feature1 = feature_map[seg[3:4]].squeeze(0)
            mask = mask[3:4].reshape(1, self.image_height, self.image_width)
        else:
            raise ValueError("feature_level=", feature_level)
        
        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
       
        return point_feature.cuda(), mask.cuda()
    
    def get_depth_map(self, depth_dir, split='train',data_type="nerfies"):
        if data_type == "nerfies":
            if split=='train':
                real_id = self.colmap_id*4 + 1
            elif split == 'test': # test
                real_id = self.colmap_id*4 + 3
            else: # video
                real_id = self.colmap_id +1
            depth_name = os.path.join(depth_dir, f"{real_id:06}")

        elif data_type == "dynerf": 
            frame_id = self.colmap_id % 300

            if split ==  'test':
                assert self.colmap_id<300
            elif split == "video":
                return None, None

            depth_name = os.path.join(depth_dir, f"{self.cam_name}-{frame_id:04}")

        elif data_type == "colmap":
            # TODO: this is hardcoded, need to fix this! has to do with how colmap generated more images that are blurred
            #  and essentially repeat the dataset every few times
            # frame_id = self.colmap_id % 80
            frame_id = self.colmap_id + 1
            # if frame_id == 0:
            #     frame_id = 80
            depth_name = os.path.join(depth_dir, f"{frame_id:06}")            
        else:
            raise NotImplementedError

        depth_map = torch.from_numpy(np.load(depth_name + ".npy"))

        return depth_map.float().cuda().unsqueeze(0) # add channel dim
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

        
def rotate_camera_around_center(camera, angle_degrees, center,axs='x'):
    """
    Rotate the camera around the center of the scene horizontally by the given angle.
    
    Args:
        camera (Camera): The camera object to be rotated.
        angle_degrees (float): The angle in degrees by which to rotate the camera.
    """
    # Convert the angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    if axs == 'x':
        # Define the rotation matrix for a horizontal rotation around the Y-axis
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ], dtype=np.float32)

        inverse_rotation_matrix = np.array([
            [np.cos(angle_radians), 0, -np.sin(angle_radians)],
            [0, 1, 0],
            [np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axs == 'y':
        # Define the rotation matrix for a horizontal rotation around the Y-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), np.sin(angle_radians)],
            [0, -np.sin(angle_radians), np.cos(angle_radians)]
        ], dtype=np.float32)

        inverse_rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ], dtype=np.float32)
        # import pdb; pdb.set_trace()
    elif axs == 'z':
        # Define the rotation matrix for a horizontal rotation around the Y-axis
        rotation_matrix = np.array([
            [np.cos(angle_radians), np.sin(angle_radians),0],
            [-np.sin(angle_radians), np.cos(angle_radians),0],
            [0,0,1]
        ], dtype=np.float32)

        inverse_rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians),0],
            [np.sin(angle_radians), np.cos(angle_radians),0],
            [0,0,1]
        ], dtype=np.float32)
    
    # Convert the rotation matrix to a torch tensor
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    
    # Ensure camera.R is a torch tensor
    if not isinstance(camera.R, torch.Tensor):
        camera.R = torch.tensor(camera.R, dtype=torch.float32)
    
    # Update the camera's rotation matrix
    camera.R = torch.matmul(rotation_matrix, camera.R)
    
    # Ensure camera.T is a torch tensor
    camera.T = torch.tensor(camera.T, dtype=torch.float32)
    
    # Center position as a torch tensor
    center_position = torch.tensor(center, dtype=torch.float32)
    
    # Translate the camera position relative to the center
    # print(camera.T)
    camera_position_relative = camera.T - center_position

    inverse_rotation_matrix = torch.tensor(inverse_rotation_matrix, dtype=torch.float32)
    # Rotate the camera position around the center
    rotated_position_relative = torch.matmul(inverse_rotation_matrix, camera_position_relative)
    
    # Translate the camera position back
    camera.T = rotated_position_relative + center_position
    # print(camera.T)
    # import pdb; pdb.set_trace()
    
    # Convert back to numpy for further processing if needed
    camera.T = camera.T.cpu().numpy()
    
    # Update the world view transform matrix
    camera.world_view_transform = torch.tensor(getWorld2View2(camera.R.cpu().numpy(), camera.T, camera.trans, camera.scale), dtype=torch.float32).transpose(0, 1)
    
    # Update the full projection transform matrix
    camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)
    
    # Update the camera center
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    
    
    return camera


