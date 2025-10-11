"""
Train Qwen feature autoencoder (inspired by splattalk)
"""

from pathlib import Path
import argparse
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoencoder.dataset import Autoencoder_dataset
from autoencoder.model_qwen import QwenAutoencoder


def train(
    clip_path: str,
    lf_dir_names: List[str] = ['qwen_patch_features', 'qwen_instance_features'],
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = 'cuda:0',
    latent_dim: int = 3,
) -> None:
    """Train Qwen feature autoencoder.

    Parameters mirror the command-line options so this can be called programmatically.
    Unused parameters (e.g., cos_weight, eval_after) are included for API compatibility.
    """
    data_dirs = [
        Path(clip_path) / i for i in lf_dir_names
    ]
    ae_dir = Path(clip_path) / 'autoencoder'
    ae_dir.mkdir(parents=True, exist_ok=True)

    dataset = Autoencoder_dataset(data_dirs)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = QwenAutoencoder(input_dim=3584, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    tb = SummaryWriter(ae_dir)

    global_step = 0
    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch.to(device, dtype=torch.float32)
            x_rec = model(x)
            loss = F.mse_loss(x_rec, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tb.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))

        # Eval
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device, dtype=torch.float32)
                x_rec = model(x)
                val_loss_sum += F.mse_loss(x_rec, x, reduction='sum').item()
        val_loss = val_loss_sum / len(val_dataset)
        tb.add_scalar('val/loss', val_loss, epoch)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ae_dir / 'best_ckpt.pth')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), ae_dir / f"{epoch+1}_ckpt.pth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--language_name', type=str, default='qwen_features')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--latent_dim', type=int, default=256)
    args = parser.parse_args()
    train(
        clip_path=args.dataset_path,
        lf_dir_names=[args.language_names],
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        latent_dim=args.latent_dim,
    )


if __name__ == '__main__':
    main()


