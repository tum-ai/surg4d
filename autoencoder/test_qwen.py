import os
import argparse
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Autoencoder_dataset
from model_qwen import QwenAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--language_name', type=str, default='qwen_features')
    parser.add_argument('--output_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--latent_dim', type=int, default=256)
    args = parser.parse_args()

    data_dir = os.path.join(args.dataset_path, args.language_name)
    if args.output_name is not None:
        output_dir = os.path.join(args.dataset_path, f"{args.language_name}-{args.output_name}_dim{args.latent_dim}")
    else:
        output_dir = os.path.join(args.dataset_path, f"{args.language_name}_dim{args.latent_dim}")
    os.makedirs(output_dir, exist_ok=True)

    # Copy segmentation maps alongside features
    for filename in os.listdir(data_dir):
        if filename.endswith('_s.npy'):
            shutil.copy(os.path.join(data_dir, filename), os.path.join(output_dir, filename))

    # Load model
    ckpt_path = f"ckpt/{args.model_name}/best_ckpt.pth"
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model = QwenAutoencoder(input_dim=3584, latent_dim=args.latent_dim).to(args.device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Dataset/Loader
    dataset = Autoencoder_dataset(data_dir)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    # Encode and collect
    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='encoding'):
            x = batch.to(args.device, dtype=torch.float32)
            z = model.encode(x)  # normalized 256-D
            all_features.append(z.detach().cpu().numpy())

    features = np.concatenate(all_features, axis=0)

    # Save back using original chunk sizes
    start = 0
    for name, rows in dataset.data_dic.items():
        out_path = os.path.join(output_dir, name + '.npy')
        np.save(out_path, features[start:start + rows])
        start += rows


if __name__ == '__main__':
    main()


