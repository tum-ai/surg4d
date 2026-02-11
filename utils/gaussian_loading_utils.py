from omegaconf import DictConfig
from loguru import logger

def get_latest_model_iteration(cfg: DictConfig):
    # Compute the last iteration for this stage based on splat config
    stage = cfg.graph_extraction.load_stage
    stage_iterations = {
        "coarse-base": cfg.splat.coarse_base_iterations,
        "coarse-lang": cfg.splat.coarse_lang_iterations,
        "fine-base": cfg.splat.fine_base_iterations,
        "fine-lang": cfg.splat.fine_lang_iterations,
    }
    last_stage_iter = stage_iterations.get(stage, 0)
    
    # Find the largest save_iteration that is <= last_stage_iter
    save_iters = sorted(cfg.splat.save_iterations)
    valid_save_iters = [it for it in save_iters if it <= last_stage_iter]
    
    if valid_save_iters:
        # Use the largest valid save iteration
        iteration = max(valid_save_iters)
        logger.info(f"Auto-selecting iteration {iteration} for stage {stage} (last stage iteration: {last_stage_iter})")
    else:
        # Fall back to last_stage_iter if no save_iterations match
        iteration = last_stage_iter
        logger.info(f"No matching save_iterations found, using last stage iteration {iteration} for stage {stage}")
    return iteration


def get_best_model_iteration(cfg: DictConfig, clip_name: str) -> int:
    stage = cfg.graph_extraction.load_stage
    iteration = -2
    logger.info(
        f"Using best checkpoint directory point_cloud/{stage}_best "
        f"for clip {clip_name}"
    )
    return iteration