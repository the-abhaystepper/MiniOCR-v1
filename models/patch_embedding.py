"""Patch embedding and 2D spatial position encoding for MiniOCR."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert a 224×224 RGB image into a grid of 196 visual tokens (14×14), each 256-dim.

    Uses a Conv2d with kernel_size=patch_size and stride=patch_size so that
    no class-token is needed: every output position directly corresponds to
    a spatial patch on the image grid.

    Shape:
        input : [B, 3, 224, 224]
        output: [B, 196, 256]
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 14
        self.num_patches = self.grid_size * self.grid_size  # 196

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Output: [B, embed_dim, grid, grid] → reshaped to [B, num_patches, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB images of shape [B, 3, 224, 224]
        Returns:
            Visual token features of shape [B, 196, 256]
        """
        B = x.shape[0]
        x = self.proj(x)            # [B, 256, 14, 14]
        x = x.flatten(2)            # [B, 256, 196]
        x = x.transpose(1, 2)       # [B, 196, 256]
        return x


class SpatialPositionEmbedding(nn.Module):
    """Explicit 2D spatial encoding that preserves the 14×14 geometry of patches.

    Rather than flattening positions into a 1D sequence, each patch receives
    a row embedding and a column embedding so the model knows *where* each
    visual token sits in the image:

        spatial_token = visual_feature + row_emb[row] + col_emb[col]

    This is critical for OCR because text has a strict 2D spatial layout.
    """

    def __init__(self, grid_size: int = 14, embed_dim: int = 256):
        super().__init__()
        self.grid_size = grid_size
        self.row_embedding = nn.Embedding(grid_size, embed_dim)
        self.col_embedding = nn.Embedding(grid_size, embed_dim)

        # Initialize as zeros so the model can learn to use them gradually
        nn.init.zeros_(self.row_embedding.weight)
        nn.init.zeros_(self.col_embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual tokens of shape [B, num_patches, embed_dim]
               (typically 196 patches from a 14×14 grid)
        Returns:
            Spatially-augmented tokens of shape [B, num_patches, embed_dim]
        """
        B, N, D = x.shape
        # Determine row/col indices for each patch position
        indices = torch.arange(N, device=x.device)  # [num_patches]
        rows = indices // self.grid_size            # row index of each patch
        cols = indices % self.grid_size             # col index of each patch

        row_emb = self.row_embedding(rows)   # [num_patches, D]
        col_emb = self.col_embedding(cols)   # [num_patches, D]

        # Broadcast to batch
        x = x + row_emb.unsqueeze(0) + col_emb.unsqueeze(0)  # [B, num_patches, D]
        return x
