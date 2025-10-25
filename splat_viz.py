from scene.gaussian_model import GaussianModel
from pathlib import Path
from extract_graph_old import init_params, load_all_models, filter_gaussians
from gaussian_renderer import render as gs_render
from gaussian_renderer import render_opacity as gs_render_opacity
from torchvision.utils import save_image
import argparse
from arguments import ModelParams, PipelineParams, ModelHiddenParams
import sys
import torch
from PIL import Image
from pathlib import Path

N_TIMESTEPS = 80
out_path = Path('splat_viz')

# ------------- input arguments ----------------------
data_path = Path('data/preprocessed_testlang/video01_00080')
output_path = Path('output/testlang/dl_False/video01_00080')
frame = 78
pc_exp_name = "pointcloud.ply"
img_exp_name = "render_img.png"

# ----------------------------------------------------
timestep = frame / (N_TIMESTEPS - 1)
language_feature_name = "qwen_cat_features_dim6"
# video_name = clip_name[:7]
# clip_prefix = clip_name[:13]

import os
# Simulate: export language_feature_hiddendim=${clip_feat_dim}
clip_feat_dim = '6'
os.environ['language_feature_hiddendim'] = clip_feat_dim
# manual argv override (copied from the setattr values)
sys.argv = [
    "splat_viz.py",
    "-s", str(data_path),
    "--language_features_name", language_feature_name,
    "--model_path", str(output_path),
    "--feature_level", "0",
    "--skip_train",
    "--skip_test",
    "--configs", "arguments/cholecseg8k/no_tv.py",
    "--mode", "lang",
    "--no_dlang", "1",
    "--load_stage", "fine-lang",
    "--num_views", "5",
    "--qwen_autoencoder_ckpt_path", str(data_path / "autoencoder/best_ckpt.pth"),
]
args, model_params, pipeline, hyperparam = init_params()
gaussians, scene, dataset = load_all_models(
    args, model_params, pipeline, hyperparam
)

# ------------- custom overrides ---------------------

# opacity_mask = (gaussians.get_opacity >= 0.5).squeeze()
# filter_gaussians(gaussians, opacity_mask)
# gaussians._features_rest = torch.zeros_like(gaussians._features_rest)
# gaussians._features_dc = gaussians._opacity.unsqueeze(-1).expand_as(gaussians._features_dc)
# gaussians._opacity = torch.full_like(gaussians._opacity, 100)

# ----------------------------------------------------


lf_transform = lambda c: c/2.0+0.5
cam = scene.getVideoCameras()[frame]
# cam.time = frame
bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
pkg = gs_render(cam, gaussians, pipeline, background, None, stage=args.load_stage, cam_type=scene.dataset_type, args=args)
rgb_render = torch.clamp(pkg["render"], 0.0, 1.0)
lf_render = torch.clamp(lf_transform(pkg["language_feature_image"]), 0.0, 1.0)
depth_render = pkg["depth"]
depth_render = (depth_render - depth_render.min()) / (depth_render.max() - depth_render.min())
opacity_render = gs_render_opacity(cam, gaussians, pipeline, background, None, stage=args.load_stage, cam_type=scene.dataset_type, args=args)
opacity_render = torch.clamp(opacity_render, 0.0, 1.0)


save_image(rgb_render, out_path / ("rgb_" + img_exp_name))
save_image(lf_render[:3], out_path / ("patch_" + img_exp_name))
save_image(lf_render[3:], out_path / ("instance_" + img_exp_name))
save_image(depth_render, out_path / ("depth_" + img_exp_name))
save_image(opacity_render, out_path / ("opacity_" + img_exp_name))

time = torch.full(
    (gaussians.get_xyz.shape[0], 1),
    float(timestep),
    device=gaussians.get_xyz.device,
    dtype=gaussians.get_xyz.dtype,
)

print(f"number of gaussians: {gaussians._xyz.shape[0]}")

xyz, scaling, rotation, _, _, _, _ = gaussians._deformation(gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, gaussians.get_language_feature, time)
gaussians._xyz, gaussians._scaling, gaussians._rotation = xyz, scaling, rotation
gaussians.save_ply(out_path / pc_exp_name)