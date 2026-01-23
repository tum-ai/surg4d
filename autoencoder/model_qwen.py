"""
Qwen feature autoencoder inspired by splattalk
"""

import torch
import torch.nn as nn


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True).clamp(min=eps)
    return x / norm


class QwenAutoencoder(nn.Module):
    """
    Autoencoder for Qwen features per paper spec:
    - Input feature dimension: 3584
    - Encoder: 3584 -> 2048 (1024) -> BN -> GeLU -> 1024 (256) -> BN -> GeLU -> 512 (32) -> BN -> GeLU -> 256 (3)
    - Decoder: 256 (3) -> 512 (64) -> GeLU -> 1024 (256) -> GeLU -> 2048 (1024) -> GeLU -> 2048 -> GeLU -> 3584
    - Latent (compressed) features are L2-normalized to a unit hypersphere
    """

    def __init__(self, input_dim: int = 3584, latent_dim: int = 3) -> None:
        super().__init__()

        # Encoder with BN + GeLU between linear layers (except after final layer)
        encoder_layers = [
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, latent_dim),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder with GeLU activations between linear layers
        decoder_layers = [
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, input_dim),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = _l2_normalize(z, dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_rec = self.decoder(z)
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec


