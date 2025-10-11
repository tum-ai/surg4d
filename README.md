# 4D LangSplatSurgery: 4D Language Gaussian Splatting via Multimodal Large Language Models on Surgery Data

## How to run on CholecSeg8k
### 1. Preprocess CholecSeg8k for Colmap
One cholecseg video sequence can be found in
```
data/cholecseg8k/video18/video18_00979
```
where 18 is the video number and `00979` is the ID of one subsequence.
You want to run
```bash
python preprocess/crop_black_borders.py --input_dir=data/cholecseg8k/video18/video18_00979
```
which crops the black camera borders, removes `_endo` from the filenames, and sorts the files into 3 newly created subdirectories:
```
data/cholecseg8k/video18/video18_00979
├── frames_cropped
├── segmentation_masks_cropped
└── watershed_masks_cropped
```

### 2. Running Colmap
```bash
python preprocess/run_nerfstudio_colmap_cpu.py \
--frames_dir data/cholecseg8k/video18/video18_00979/frames_cropped \ 
--out_dir data/cholecseg8k/video18/video18_00979/nerfstudio_colmap \
--colmap_bin /home/tumai/miniconda3/envs/4DSplat/bin/colmap
```

## Setup
4D LangSplat uses the following software versions:
- Python 3.10
- CUDA 12.4
- GCC 10.2.0

### Option 1: Using Pixi (Recommended)
[Pixi](https://pixi.sh) provides a reproducible environment with all dependencies managed automatically.

```bash
# Install pixi if you haven't already (see https://pixi.sh)
curl -fsSL https://pixi.sh/install.sh | bash

# Install all dependencies
pixi install

# Build and install all CUDA extensions and additional packages
# This installs: simple-knn, 4d-langsplat-rasterization, deva, segment-anything
pixi run setup

# Run any command in the environment
pixi run python train.py --config configs/...
```

You can also install individual components:
```bash
pixi run install-simple-knn           # Simple KNN CUDA extension
pixi run install-rasterization        # Gaussian rasterization CUDA extension
pixi run install-deva                 # DEVA tracking package
pixi run install-segment-anything     # Segment Anything from Meta
```

### Option 2: Using Conda/Pip (Traditional)
```bash
conda create -n 4DLangSplat python=3.10
conda activate 4DLangSplat
pip install -r requirements.txt
### submodules for gaussian rasterization ###
pip install -e submodules/simple-knn
pip install -e submodules/4d-langsplat-rasterization
### submodules for generate segmentation map ###
pip install -e submodules/4d-langsplat-tracking-anything-with-deva
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Prepare Datasets
Our models are trained and evaluated on [HyperNeRF](https://github.com/google/hypernerf) and [Neu3D](https://github.com/facebookresearch/Neural_3D_Video) datasets. Please follow their instructions to prepare your dataset, or run the following commands:
```bash
bash scripts/download_hypernerf.sh data/hypernerf
bash scripts/download_neu3d.sh data/neu3d
```

To evaluate the rendering results, we use [RoboFlow](https://roboflow.com/) to annotate the datasets. The annotations can be accessed through this link: [Download the Annotations](https://drive.google.com/drive/folders/1C-ciHn38vVd47TMkx2-93EUpI0z4ZdZW?usp=sharing). \
Follow [4DGaussians](https://github.com/hustvl/4DGaussians), we use COLMAP to generate the point clouds. Please follow their pipeline, or use ours: [Download the Point Clouds](https://drive.google.com/drive/folders/1_JOObfpXrCq3v_NYKwDt6vRHIbb0oVek?usp=sharing)

Then put them under `data/<hypernerf or neu3d>/<dataset name>`. You need to ensure that the data folder is organized as follows:
```
|——data
|   | hypernerf
|       | americano
|           |——annotations
|               |——train
|               |——README
|               |——video_annotations.json
|           |——camera
|           |——rgb
|               |——1x
|                   |——000001.png
|                   ...
|               |——2x        
|               ...
|           |——dataset.json
|           |——metadata.json
|           |——points.npy
|           |——scene.json
|           |——points3D_downsample2.ply
|       |——chickchicken
|       ...
|   | neu3d
|       | coffee_martini
|           |——annotations
|               |——train
|               |——README
|           |——cam00
|               |——images
|                   |——0000.png
|                   ...
|           |——cam01
|           ...
|           |——cam00.mp4
|           |——cam01.mp4
|           ...
|           |——poses_bounds.npy
|           |——points3D_downsample2.ply
|      |——cur_roasted_beef
|      ...
```

## QuickStart
We provide the pretrained checkpoints of gaussian model and autoencoder: [Download Pretrained Checkpoint](https://drive.google.com/drive/folders/1-G8I5cJCD66fjpvejUzF9QPRJU_GNxj0?usp=sharing).

For HyperNeRF dataset, take `americano` as an example. Put checkpoint folder upder the  `output/hypernerf/americano` and run the following commands for rendering and evaluation
```bash
bash scripts/render-hypernerf.sh
bash scripts/eval-hypernerf.sh
```
For Neu3D dataset, take `coffee_martini` as an example. Put checkpoint folder under the  `output/neu3d/coffee_martini` and run the following commands for rendering and evaluation
```bash
bash scripts/render-neu3d.sh
bash scripts/eval-neu3d.sh
```

The evaluation results will be saved under `eval/eval_results`.

## Training Guide
### Step 1: Generate Segmentation Map using DEVA
First execute the demo script to generate segmentation maps:
```bash
cd submodules/4d-langsplat-tracking-anything-with-deva
bash scripts/download_models.sh # Download the model parameters if you are a first time user 
bash scripts/demo-chickchicken.sh
```
The output segmentation maps will be saved in `submodules/4d-langsplat-tracking-anything-with-deva/output`

### Step 2: Extract CLIP and Video Features
Extract CLIP features:
```bash
bash scripts/extract_clip_features.sh
```
Generate video features:
```bash
bash scripts/generate-video-feature.sh
```
These commands will create two feature directories under your dataset path:
- `clip_features`: Extracted by CLIP model
- `video_features`: Extracted by E5 model

### Step 3: Train and Evaluate 4D LangSplat
Run the training and evaluation script:
```bash
bash scripts/train_eval.sh
```
This will train the 4D LangSplat field and perform evaluation.

## Reference

This repository contains the official forked implementation of the paper "4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models" (CVPR 2025).
## BibTeX
```
@inproceedings{li20254dlangsplat4dlanguage,
    title={4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models}, 
    author={Wanhua Li and Renping Zhou and Jiawei Zhou and Yingwei Song and Johannes Herter and Minghan Qin and Gao Huang and Hanspeter Pfister},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```

| [Project page](https://4d-langsplat.github.io) | [Full Paper](https://arxiv.org/abs/2503.10437) | [Video](https://youtu.be/L2OzQ91eRG4) |\
| Datasets Annotations | [Google Drive](https://drive.google.com/drive/folders/1C-ciHn38vVd47TMkx2-93EUpI0z4ZdZW?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/1ZMOk0UFQ39WJ7TtTXy9gkA?pwd=g9rg)\
| Pretrained Model | [Google Drive](https://drive.google.com/drive/folders/1-G8I5cJCD66fjpvejUzF9QPRJU_GNxj0?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/1TmBW1ZjZfjLQTGxpDXZzlg?pwd=3kmw)\
| Pregenerated Point Clouds by COLMAP | [Google Drive](https://drive.google.com/drive/folders/1_JOObfpXrCq3v_NYKwDt6vRHIbb0oVek?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/15jDvS-zSW7pfdvzdwP32mQ?pwd=9y2u)

## Setting Up Individual GitHub Access on Shared Server

### 1. Generate Your Personal SSH Key
Each person should generate their own SSH key with a unique identifier:
```bash
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/id_ed25519_yourname
```

### 2. Add Public Key to GitHub
1. Copy your public key content:
   ```bash
   cat ~/.ssh/id_ed25519_yourname.pub
   ```
2. Go to GitHub → Settings → SSH and GPG keys → New SSH key
3. Paste the public key content and save

### 3. Test Your SSH Connection
Test the connection using your alias (we'll set this up in step 4):
```bash
ssh -T git@github-yourname
```

### 4. Configure SSH Alias
Add your personal SSH configuration block to `~/.ssh/config`:
```
Host github-YOURNAME
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_YOURNAME
```
Replace `YOURNAME` with your actual identifier (same as used in the key filename).

### 5. Clone Repositories Using Your Alias
When cloning repositories, use your personal alias:
```bash
git clone --recursive git@github-yourname:username/repository-name.git
```

### 6. Set Git Identity Per Repository
Inside each cloned repository, set your personal Git identity:
```bash
cd repository-name
git config user.name "Your Full Name"
git config user.email "your-email@example.com"
```

### 7. Set Up Shared Data Symlinks
To avoid cluttering the remote server and simplify access to shared data and results, create symlinks to the shared data repository:
```bash
cd surgery-scene-graphs
ln -s ~/shared_data/4DLangSplatSurgery/data ./data
ln -s ~/shared_data/4DLangSplatSurgery/output ./output
ln -s ~/shared_data/4DLangSplatSurgery/autoencoder/ckpt ./autoencoder/ckpt
```

This allows everyone to use their local repository folders while accessing the same shared datasets, model outputs, and checkpoints.

This setup ensures that each person's commits are properly attributed to them while working on the shared server environment.
