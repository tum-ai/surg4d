import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import gc

# Import from extract_graphs.py
from extract_graphs import load_gaussian_model

def setup_log_yaxis():
    ax = plt.gca()
    # Major ticks at every power of 10
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    # Minor ticks at 2, 3, 4, 5, 6, 7, 8, 9 * power of 10
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=15))
    # Major labels with scientific notation
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10.0))
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Setup temp output directory
    temp_dir = Path("output/temp_opacities")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    all_opacities = []
    
    print(f"Starting opacity dump for {len(cfg.clips)} clips...")
    
    for clip in tqdm(cfg.clips, desc="Processing clips"):
        try:
            # Load model like in extract_graphs.py
            # Note: extract_graphs.py expects some environment variables which load_gaussian_model sets internally
            gaussians, scene, dataset, args, pipeline = load_gaussian_model(clip, cfg)
            
            # Extract opacities
            opacities = gaussians.get_opacity.squeeze().detach().cpu().numpy()
            all_opacities.append(opacities)
            
            # Individual histogram for each clip
            plt.figure(figsize=(10, 6))
            plt.hist(opacities, bins=100, edgecolor="black", alpha=0.7)
            plt.xlabel("Opacity")
            plt.ylabel("Count")
            plt.yscale('log')  # Use log scale as opacities are often skewed towards extremes
            setup_log_yaxis()
            plt.title(f"Gaussian Opacity Distribution - {clip.name}\n({len(opacities)} gaussians)")
            
            save_path = temp_dir / f"{clip.name}_opacity.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            # Cleanup GPU memory
            del gaussians
            del scene
            del dataset
            del args
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing {clip.name}: {e}")
            continue

    if all_opacities:
        # Global histogram over all scenes
        global_opacities = np.concatenate(all_opacities)
        
        plt.figure(figsize=(12, 8))
        plt.hist(global_opacities, bins=200, edgecolor="black", alpha=0.7)
        plt.xlabel("Opacity")
        plt.ylabel("Count")
        plt.yscale('log')
        setup_log_yaxis()
        plt.title(f"Global Gaussian Opacity Distribution\n({len(global_opacities)} total gaussians over {len(all_opacities)} clips)")
        
        global_save_path = temp_dir / "global_opacity_histogram.png"
        plt.savefig(global_save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"\nSuccessfully processed {len(all_opacities)} clips.")
        print(f"Global opacity count: {len(global_opacities)}")
        print(f"Histograms saved to: {temp_dir.absolute()}")
    else:
        print("\nNo opacities were collected. Check if models exist and paths are correct.")

if __name__ == "__main__":
    main()
