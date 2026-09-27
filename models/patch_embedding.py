"""Patch embedding and 2D spatial position encoding."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split a 224×224 RGB image into 196 non-overlapping 16×16 patch tokens."""

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                # [B, embed_dim, 14, 14]
        x = x.flatten(2)                # [B, embed_dim, 196]
        x = x.transpose(1, 2)           # [B, 196, embed_dim]
        return x


class SpatialPositionEmbedding(nn.Module):
    """Learned row + column positional embeddings for 2D patch layout."""

    def __init__(self, grid_size: int = 14, embed_dim: int = 256):
        super().__init__()
        self.grid_size = grid_size
        self.row_embedding = nn.Embedding(grid_size, embed_dim)
        self.col_embedding = nn.Embedding(grid_size, embed_dim)
        nn.init.zeros_(self.row_embedding.weight)
        nn.init.zeros_(self.col_embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        indices = torch.arange(N, device=x.device)
        rows = indices // self.grid_size
        cols = indices % self.grid_size
        x = x + self.row_embedding(rows).unsqueeze(0)
        x = x + self.col_embedding(cols).unsqueeze(0)
        return x
