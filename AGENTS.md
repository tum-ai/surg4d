# Agent Guide: Surgery Scene Graphs

This repository implements a 4D Language Splatting pipeline for surgery scene graph extraction, specifically targeting surgical datasets like Cholec80 and CholecSeg8k.

## 🛠 Build & Run Commands

All commands MUST be prefixed with `pixi run python` to ensure the correct environment and dependencies (including CUDA extensions) are loaded.

### 1. Environment Setup
```bash
pixi install
pixi run setup  # CRITICAL: Installs CUDA extensions and submodules
```

### 2. Pipeline Execution
The pipeline is managed via Hydra. Configs are located in `conf/`.

| Step | Command | Config Section | Description |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | `pixi run python preprocess.py` | `preprocessing` | Extracts frames, depth, and masks. |
| **Feature Gen** | `pixi run python generate_qwen_features.py` | `feature_extraction` | Extracts Qwen-VL features per frame. |
| **Autoencoders** | `pixi run python train_autoencoders.py` | `autoencoder` | Trains latent compression for features. |
| **Splat Training** | `pixi run python train_splats.py` | `splat` | Trains 4D Gaussian Splats. |
| **Graph Extraction**| `pixi run python extract_graphs.py` | `graph_extraction` | Extracts scene graphs from splats. |
| **Evaluation** | `pixi run python evaluate_benchmark.py` | `eval` | Runs benchmark evaluation. |
| **Metrics** | `pixi run python compute_metrics.py` | `compute_metrics` | Computes final metric scores. |

To run the full pipeline for all clips defined in `conf/clips/`:
```bash
pixi run python run_pipeline.py
```

### 3. GPU Compute (Slurm)
The development node typically lacks a GPU. Training and rendering tasks should be submitted to the Slurm cluster.
**Slurm Template (`start.sh`):**
```bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
pixi run python train_splats.py
```
Submit with: `sbatch start.sh`

---

## 🎨 Code Style & Guidelines

### Core Philosophy: "Fail Fast, No Defensiveness"
- **No Try-Except for Logic:** If a tensor shape is wrong or a file is missing, let it crash. ML bugs are harder to find if hidden by fallbacks.
- **Hydra is Law:** Never hardcode defaults or provide fallbacks for config values in Python code. If it's configurable, it belongs in `conf/`.
- **Pipeline Wiring:** Use Hydra's interpolation `${clips.some_val}` to share state between pipeline steps. Each script should ideally only read from its own config group.

### Python Conventions
- **Naming:**
    - Classes: `PascalCase` (e.g., `GaussianModel`, `Scene`).
    - Functions/Variables: `snake_case`.
    - Constants: `UPPER_SNAKE_CASE`.
- **Imports Order:**
    1. Standard libs (`os`, `sys`, `typing`).
    2. Data/Math libs (`numpy`, `torch`, `scipy`).
    3. Local modules (`from utils import ...`).
- **Typing:**
    - Mandatory for public function signatures.
    - Use `jaxtyping` for tensors: `Float[Tensor, "B N D"]`.
    - Use `typing.Literal` for mode flags.
- **Logging:**
    - Use `loguru` (e.g., `from loguru import logger`). Avoid `print()`.
- **WandB:**
    - Integration is enabled via `wandb: t` environment variable or config.

### Project Structure
- `conf/`: Hydra configuration files organized by pipeline step.
- `scene/`: Core Gaussian Splatting and data loading logic.
- `utils/`: Helper functions (math, camera, rendering, etc.).
- `submodules/`: External dependencies (rasterizer, depth-anything, etc.).

### Data Structure
Expected root directories for datasets:
- `data/cholec80`: Raw video data.
- `data/cholecseg8k`: Segmentation labels.
- `data/cholect50`: Temporal annotations.
- `data/preprocessed`: Output of the preprocessing step.

---

## 🤖 AI / Cursor Instruction Set

When operating in this repository, you MUST:

1.  **Always use `pixi run python`** for any execution.
2.  **Verify Setup:** If CUDA-related imports fail, ensure `pixi run setup` was executed.
3.  **Config-First:** When adding parameters, add them to `conf/config.yaml` or the relevant sub-config instead of hardcoding.
4.  **No GPU Assumptions:** Always assume the current node is a CPU-only head node unless `srun` or `sbatch` is being used.
5.  **Fail Fast:** Do not add defensive `getattr(cfg, 'param', default)` patterns. Assume the config is complete.
6.  **Traceability:** Ensure all new scripts are compatible with the `run_pipeline.py` workflow.
7.  **Hydra Wiring:** If a script needs a path from a previous step, use `${clips.output_dir}` or similar interpolation in the config rather than hardcoding.
8.  **Single Test Run:** To run a specific script or part of the pipeline for debugging, use `pixi run python <script>.py clips=<clip_name>`.
9.  **No Chitchat:** Keep responses concise and focused on the code or task at hand.
